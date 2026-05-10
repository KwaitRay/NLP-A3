"""Dense retrieval over the evidence corpus using sentence-transformers + FAISS.

Default embedder: ``BAAI/bge-m3`` (568M, 1024-d, multilingual). Falls back to
``BAAI/bge-small-en-v1.5`` (33M, 384-d) if VRAM is tight or speed matters.

Encoding the full ~1.2M corpus is a ~30-60 min one-off on Colab T4. Always
persist the embeddings + FAISS index to Drive so subsequent sessions skip it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# Heavy deps are imported inside methods.


DEFAULT_MODEL = "BAAI/bge-m3"
LIGHT_MODEL = "BAAI/bge-small-en-v1.5"


class DenseRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        max_seq_length: int = 256,
        fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self.fp16 = fp16
        self._model = None
        self._index = None
        self._ev_ids: list[str] = []
        self._dim: int | None = None

    # -- build ---------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from ..paths import resolve_model_path
            # Note: bge-m3 ships only pytorch_model.bin (no safetensors). Loading
            # via torch.load() requires either torch>=2.6 OR transformers<4.50;
            # see environment_setting.md Q8 / requirements.txt pin.
            # resolve_model_path() returns models/<basename>/ if pre-downloaded
            # via scripts.download_models, else the original HF repo_id.
            path = resolve_model_path(self.model_name)
            self._model = SentenceTransformer(path, device=self.device)
            self._model.max_seq_length = self.max_seq_length
            if self.fp16 and self.device.startswith("cuda"):
                self._model.half()
        return self._model

    def build(
        self,
        evidence_corpus: dict[str, str],
        *,
        save_dir: str | Path,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        import faiss
        import gc
        import numpy as np

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._ev_ids = list(evidence_corpus.keys())

        # Encode in chunks, persist each chunk so a session crash doesn't lose
        # everything. After the loop, stream-add into FAISS to avoid a 2x peak
        # RAM spike from np.concatenate (which would OOM on 12GB Colab).
        chunk_size = batch_size * 200  # ~6.4k passages per chunk at bs=32
        chunks_dir = save_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        n = len(self._ev_ids)
        n_chunks = (n + chunk_size - 1) // chunk_size

        all_chunks_done = all(
            (chunks_dir / f"emb_{ci:05d}.npy").exists() for ci in range(n_chunks)
        )

        if not all_chunks_done:
            # Only load the model + texts list when there's encoding left to do.
            model = self._load_model()
            texts = [evidence_corpus[eid] for eid in self._ev_ids]
            try:
                import torch
                _torch = torch
            except ImportError:
                _torch = None

            for ci in range(n_chunks):
                chunk_path = chunks_dir / f"emb_{ci:05d}.npy"
                if chunk_path.exists():
                    continue
                sl = slice(ci * chunk_size, min((ci + 1) * chunk_size, n))
                emb = model.encode(
                    texts[sl],
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                ).astype("float32")
                np.save(chunk_path, emb)
                if _torch is not None and _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

            del texts
            gc.collect()

        # Stream chunks into FAISS one at a time. Peak RAM = single chunk
        # (~26MB) + growing index (up to ~5GB at end), vs. the old concat path
        # which held ~10GB transiently.
        print(f"Finalising FAISS IndexFlatIP from {n_chunks} chunks (streaming)...")
        first = np.load(chunks_dir / "emb_00000.npy")
        self._dim = int(first.shape[1])
        index = faiss.IndexFlatIP(self._dim)
        index.add(np.ascontiguousarray(first, dtype="float32"))
        del first
        gc.collect()

        for ci in range(1, n_chunks):
            chunk = np.load(chunks_dir / f"emb_{ci:05d}.npy")
            index.add(np.ascontiguousarray(chunk, dtype="float32"))
            del chunk
            if (ci + 1) % 20 == 0:
                gc.collect()
                print(f"  added {ci+1}/{n_chunks} chunks, ntotal={index.ntotal}")

        self._index = index
        print(f"FAISS built: ntotal={index.ntotal}, dim={self._dim}")

        # Persist
        faiss.write_index(index, str(save_dir / "faiss.index"))
        (save_dir / "ev_ids.txt").write_text("\n".join(self._ev_ids), encoding="utf-8")
        (save_dir / "meta.json").write_text(
            json.dumps({"model": self.model_name, "dim": self._dim, "n": n, "normalize": normalize}),
            encoding="utf-8",
        )
        print(f"Persisted to {save_dir}")

    @classmethod
    def load(
        cls,
        save_dir: str | Path,
        device: str = "cuda",
        max_seq_length: int = 256,
        fp16: bool = True,
    ) -> "DenseRetriever":
        import faiss

        save_dir = Path(save_dir)
        meta = json.loads((save_dir / "meta.json").read_text(encoding="utf-8"))
        inst = cls(
            model_name=meta["model"],
            device=device,
            max_seq_length=max_seq_length,
            fp16=fp16,
        )
        inst._ev_ids = (save_dir / "ev_ids.txt").read_text(encoding="utf-8").splitlines()
        inst._dim = meta["dim"]
        inst._index = faiss.read_index(str(save_dir / "faiss.index"))
        return inst

    # -- query ---------------------------------------------------------------

    def search(self, query: str, k: int = 200) -> list[tuple[str, float]]:
        return self.search_batch([query], k=k)[0]

    def search_batch(self, queries: Iterable[str], k: int = 200) -> list[list[tuple[str, float]]]:
        if self._index is None:
            raise RuntimeError("retriever not built/loaded")
        import numpy as np

        model = self._load_model()
        qs = list(queries)
        q_emb = model.encode(
            qs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, idxs = self._index.search(q_emb, k)
        out: list[list[tuple[str, float]]] = []
        for i in range(len(qs)):
            out.append([
                (self._ev_ids[j], float(s))
                for j, s in zip(idxs[i].tolist(), scores[i].tolist())
                if j >= 0
            ])
        return out
