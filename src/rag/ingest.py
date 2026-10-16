"""Build (or rebuild) the document search index used by src.rag.ask.

    python -m src.rag.ingest                                   # README.md, data/README.md, knowledge/
    python -m src.rag.ingest --source knowledge --source docs\\hvac_manual.pdf

src.rag.ask rebuilds the index by itself when documents change; run this to check what gets indexed.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from ..config import KNOWLEDGE_DIR, RAG_INDEX_DIR
from ..errors import PowerPulseError
from .documents import DEFAULT_MAX_WORDS
from .index import default_sources, ensure_index


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Index documents for the PowerPulse question-answering assistant.")
    p.add_argument("--source", action="append",
                   help="File or folder to index (repeatable). Default: README.md, data/README.md, knowledge/.")
    p.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS, help="Words per chunk (default 200).")
    p.add_argument("--index-dir", default=str(RAG_INDEX_DIR))
    args = p.parse_args(argv)
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    try:
        index, _ = ensure_index(args.source or default_sources(), args.index_dir, args.max_words, rebuild=True)
    except PowerPulseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    per_source = Counter(c.source for c in index.chunks)
    for source, n in sorted(per_source.items()):
        print(f"{n:5d} chunks  {source}")
    print(f"Indexed {len(index.chunks)} chunks from {len(per_source)} documents -> {args.index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
