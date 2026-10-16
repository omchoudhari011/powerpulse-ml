"""TF-IDF search index over document chunks.

TF-IDF (scikit-learn, already a dependency) matches on words, runs offline and is instant for a few
hundred documents. The index stores a fingerprint (path + modification time) of every source file and
is rebuilt automatically when documents are added, changed or removed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import DATA_DIR, KNOWLEDGE_DIR, PROJECT_ROOT, RAG_INDEX_DIR
from ..errors import RAGError
from .documents import DEFAULT_MAX_WORDS, Chunk, discover_documents, display_path, load_chunks

INDEX_FILE = "index.joblib"
INFO_FILE = "index_info.json"


def default_sources() -> List[Path]:
    return [PROJECT_ROOT / "README.md", DATA_DIR / "README.md", KNOWLEDGE_DIR]


def fingerprint(paths: Iterable) -> Dict[str, int]:
    return {display_path(f): f.stat().st_mtime_ns for f in discover_documents(paths)}


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class KnowledgeIndex:
    def __init__(self, vectorizer, matrix, chunks: List[Chunk], roots: List[str], sources: Dict[str, int],
                 max_words: int, built_at: str):
        self.vectorizer, self.matrix, self.chunks = vectorizer, matrix, chunks
        self.roots = roots          # the files/folders it was built from, so a rebuild uses the same ones
        self.sources, self.max_words, self.built_at = sources, max_words, built_at

    @classmethod
    def build(cls, paths: Iterable, max_words: int = DEFAULT_MAX_WORDS) -> "KnowledgeIndex":
        paths = [Path(p).resolve() for p in paths]
        chunks = load_chunks(paths, max_words)
        if not chunks:
            raise RAGError("No documents with text found to index. Add .md, .txt or .pdf files to "
                           f"{display_path(KNOWLEDGE_DIR)}/.")
        # The heading is indexed with its text so a question about "setup" finds the Setup section.
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        try:
            matrix = vectorizer.fit_transform(f"{c.section}\n{c.text}" for c in chunks)
        except ValueError as exc:  # empty vocabulary, e.g. documents of stop words only
            raise RAGError(f"Documents contain no searchable words: {exc}") from exc
        return cls(vectorizer, matrix, chunks, [str(p) for p in paths], fingerprint(paths), max_words,
                   datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def search(self, query: str, k: int = 5, min_score: float = 0.01) -> List[SearchResult]:
        """Best k chunks by cosine similarity (TF-IDF rows are L2-normalised, so a dot product)."""
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]

    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / INDEX_FILE)
        info = {"built_at": self.built_at, "n_chunks": len(self.chunks), "max_words": self.max_words,
                "roots": self.roots, "sources": sorted(self.sources)}
        (directory / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory) -> "KnowledgeIndex":
        path = Path(directory) / INDEX_FILE
        if not path.exists():
            raise RAGError(f"No index at {path}; run python -m src.rag.ingest first.")
        return joblib.load(path)


def ensure_index(paths: Optional[Iterable] = None, index_dir=None, max_words: int = DEFAULT_MAX_WORDS,
                 rebuild: bool = False) -> Tuple[KnowledgeIndex, bool]:
    """Load the saved index, rebuilding it when forced, missing or out of date. Returns (index, rebuilt).

    With ``paths=None`` the saved index's own sources are reused (default_sources() when there is none),
    so an index built with ``ingest --source ...`` keeps those sources.
    """
    index_dir = Path(index_dir or RAG_INDEX_DIR)
    saved = None
    if (index_dir / INDEX_FILE).exists():
        try:
            saved = KnowledgeIndex.load(index_dir)
        except Exception:  # unreadable/incompatible pickle: rebuild below
            saved = None
    if paths is None:
        paths = saved.roots if saved is not None else default_sources()
    paths = [Path(p).resolve() for p in paths]
    if (not rebuild and saved is not None and saved.max_words == max_words
            and saved.roots == [str(p) for p in paths] and saved.sources == fingerprint(paths)):
        return saved, False
    index = KnowledgeIndex.build(paths, max_words)
    index.save(index_dir)
    return index, True
