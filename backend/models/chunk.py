from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    page: int
    text: str
    section_hint: str | None = None

