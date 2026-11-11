import re
from dataclasses import dataclass
from pathlib import Path

from .config import CORPUS_DIR

MAX_CHARS = 1500
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    file: str
    drug: str          # e.g. "warfarin" (or "glossary")
    source_type: str   # "dailymed" | "medlineplus" | "glossary"
    section: str       # heading path, e.g. "Warfarin > Contraindications"
    text: str

    @property
    def embed_text(self) -> str:
        # Prefixing context makes retrieval much more accurate.
        return f"{self.drug} ({self.source_type}) - {self.section}\n{self.text}"


def parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem.lower()
    if stem == "glossary":
        return "glossary", "glossary"
    source_type, _, drug = stem.partition("-")
    return source_type, drug


def _split_long(text: str) -> list[str]:
    """Split on paragraph boundaries into pieces of at most MAX_CHARS."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) + 2 > MAX_CHARS:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_file(path: Path) -> list[Chunk]:
    source_type, drug = parse_filename(path)
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            title = " > ".join(t for _, t in stack) or path.stem
            sections.append((title, body))
        buf.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            buf.append(line)
    flush()

    chunks = []
    for title, body in sections:
        for piece in _split_long(body):
            chunks.append(Chunk(path.name, drug, source_type, title, piece))
    return chunks


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            chunks.extend(chunk_file(path))
    return chunks


if __name__ == "__main__":
    all_chunks = load_corpus()
    print(f"{len(all_chunks)} chunks from {len({c.file for c in all_chunks})} files\n")
    for name in ("dailymed-warfarin.md", "dailymed-metformin.md"):
        print(name)
        for c in all_chunks:
            if c.file == name:
                print(f"  - {c.section}  ({len(c.text)} chars)")
        print()