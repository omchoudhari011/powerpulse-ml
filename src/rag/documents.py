"""Load knowledge documents and split them into chunks small enough to retrieve precisely.

Markdown is split at headings (headings inside ``` code fences are ignored), PDFs per page, plain text
as one block. Each block is then packed paragraph by paragraph into chunks of at most ``max_words``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from ..config import PROJECT_ROOT
from ..errors import RAGError

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")
DEFAULT_MAX_WORDS = 200
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    source: str     # path relative to the project root when possible
    section: str    # nearest Markdown heading or "page N" for PDFs; "" when unknown
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} > {self.section}" if self.section else self.source


def display_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def discover_documents(paths: Iterable) -> List[Path]:
    """Expand files and directories (recursively) into a sorted list of supported documents."""
    found = set()
    for p in map(Path, paths):
        if p.is_dir():
            found.update(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
                         and not any(part.startswith(".") for part in f.relative_to(p).parts))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            found.add(p)
    return sorted(f.resolve() for f in found)


def read_blocks(path: Path) -> List[Tuple[str, str]]:
    """Return (section, text) blocks for one document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RAGError(f"Reading {path.name} needs pypdf: pip install pypdf") from exc
        try:
            pages = PdfReader(str(path)).pages
            return [(f"page {i}", page.extract_text() or "") for i, page in enumerate(pages, 1)]
        except Exception as exc:  # pypdf raises many types for damaged/encrypted files
            raise RAGError(f"Could not read {path.name}: {exc}") from exc
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix != ".md":
        return [("", text)]
    blocks, section, lines, in_fence = [], "", [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            blocks.append((section, "\n".join(lines)))
            section, lines = match.group(2).strip(), []
        else:
            lines.append(line)
    blocks.append((section, "\n".join(lines)))
    return blocks


def split_text(text: str, max_words: int = DEFAULT_MAX_WORDS) -> List[str]:
    """Pack paragraphs into chunks of <= max_words; a longer paragraph is cut by words."""
    if max_words < 20:
        raise RAGError("max_words must be at least 20.")
    chunks, current, count = [], [], 0

    def flush():
        nonlocal current, count
        if current:
            chunks.append("\n\n".join(current))
        current, count = [], 0

    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        n = len(para.split())
        if n > max_words:
            flush()
            words = para.split()
            chunks.extend(" ".join(words[i:i + max_words]) for i in range(0, n, max_words))
            continue
        if count + n > max_words:
            flush()
        current.append(para)
        count += n
    flush()
    return chunks


def load_chunks(paths: Iterable, max_words: int = DEFAULT_MAX_WORDS) -> List[Chunk]:
    chunks = []
    for path in discover_documents(paths):
        source = display_path(path)
        for section, text in read_blocks(path):
            chunks.extend(Chunk(source, section, piece) for piece in split_text(text, max_words))
    return chunks
