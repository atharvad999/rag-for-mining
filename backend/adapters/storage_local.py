from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class LocalStorage:
    """Simple local filesystem storage for dev mode.

    Files are saved under a configured root directory (e.g., backend/data/storage).
    Paths are relative IDs like "tenders/<uuid>_file.pdf".
    """

    def __init__(self, root: str):
        self.root = root
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def upload_file(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        dest = Path(self.root) / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return str(path)

    def download_file(self, path: str) -> bytes:
        fp = Path(self.root) / path
        if not fp.exists():
            raise FileNotFoundError(f"Local object not found: {path}")
        return fp.read_bytes()

    def public_url(self, path: str) -> Optional[str]:
        # Not applicable for local dev; return a file:// URL for convenience
        return f"file://{os.path.abspath(os.path.join(self.root, path))}"

