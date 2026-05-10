"""BM25 retrieval over the evidence corpus.

Runs on Colab (and locally if ``bm25s`` is installed). Persists the index to
disk so subsequent sessions reload it instantly instead of rebuilding.

Usage::

    from src.retrieval.bm25 import BM25Retriever
    r = BM25Retriever()
    r.build(evidence_corpus, save_dir="outputs/bm25_index")
    # next session
    r = BM25Retriever.load("outputs/bm25_index")
    hits = r.search("South Australia electricity prices", k=200)  # [(ev_id, score), ...]
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

# bm25s and Stemmer imported lazily inside methods so missing deps don't
# break local imports. Same pattern used in dense.py / rerank.py.


def _tokenize(text: str) -> list[str]:
    """Lower-case + simple punctuation strip + whitespace split.

    bm25s ships with its own tokenizer + stemmer, but we use a thin wrapper
    so the same tokenisation can be applied to spaCy-extracted ``important_kwd``
    fields when we add per-field weighting later.
    """
    import re
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 1]


class BM25Retriever:
    """Thin wrapper over ``bm25s`` with metadata tracking and on-disk caching."""

    def __init__(self) -> None:
        self._index = None
        self._ev_ids: list[str] = []
        self._stemmer = None

    # -- build ---------------------------------------------------------------

    def build(
        self,
        evidence_corpus: dict[str, str],
        *,
        save_dir: str | Path | None = None,
        use_stemmer: bool = True,
    ) -> None:
        import bm25s

        self._ev_ids = list(evidence_corpus.keys())
        if use_stemmer:
            try:
                import Stemmer  # type: ignore
                self._stemmer = Stemmer.Stemmer("english")
            except Exception:
                self._stemmer = None
        texts = [evidence_corpus[eid] for eid in self._ev_ids]
        token_streams = bm25s.tokenize(texts, stopwords="en", stemmer=self._stemmer)

        self._index = bm25s.BM25()
        self._index.index(token_streams)

        if save_dir:
            self.save(save_dir)

    def save(self, save_dir: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("call build() before save()")
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        self._index.save(str(p / "bm25"))
        (p / "ev_ids.txt").write_text("\n".join(self._ev_ids), encoding="utf-8")

    @classmethod
    def load(cls, save_dir: str | Path) -> "BM25Retriever":
        import bm25s

        p = Path(save_dir)
        inst = cls()
        inst._index = bm25s.BM25.load(str(p / "bm25"), load_corpus=False)
        inst._ev_ids = (p / "ev_ids.txt").read_text(encoding="utf-8").splitlines()
        try:
            import Stemmer  # type: ignore
            inst._stemmer = Stemmer.Stemmer("english")
        except Exception:
            inst._stemmer = None
        return inst

    # -- query ---------------------------------------------------------------

    def search(self, query: str, k: int = 200) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("retriever not built/loaded")
        import bm25s

        toks = bm25s.tokenize([query], stopwords="en", stemmer=self._stemmer)
        results, scores = self._index.retrieve(toks, k=k)
        # results: shape (1, k) of corpus indices; scores same shape.
        idx_list = results[0].tolist()
        sc_list = scores[0].tolist()
        return [(self._ev_ids[i], float(s)) for i, s in zip(idx_list, sc_list)]

    def search_batch(self, queries: Iterable[str], k: int = 200) -> list[list[tuple[str, float]]]:
        if self._index is None:
            raise RuntimeError("retriever not built/loaded")
        import bm25s

        qs = list(queries)
        toks = bm25s.tokenize(qs, stopwords="en", stemmer=self._stemmer)
        results, scores = self._index.retrieve(toks, k=k)
        out: list[list[tuple[str, float]]] = []
        for i in range(len(qs)):
            idx_list = results[i].tolist()
            sc_list = scores[i].tolist()
            out.append([(self._ev_ids[j], float(s)) for j, s in zip(idx_list, sc_list)])
        return out
