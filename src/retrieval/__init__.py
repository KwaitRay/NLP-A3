"""Stage 1 retrieval pipeline: BM25 + dense + fusion + rerank.

Each submodule keeps its heavy deps (bm25s, sentence-transformers, torch)
import-local so importing the package itself doesn't fail when those libs
are missing — useful during local Stage 0 development.
"""
