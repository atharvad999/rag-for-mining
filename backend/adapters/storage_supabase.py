from typing import Optional
import io

try:
    from supabase import create_client, Client
except Exception:  # pragma: no cover - keep import soft for scaffold
    create_client = None
    Client = object  # type: ignore


class SupabaseStorage:
    def __init__(self, url: str, service_role_key: str, bucket: str):
        self.url = url
        self.service_role_key = service_role_key
        self.bucket = bucket
        self.client: Optional[Client] = None
        if create_client is not None:
            try:
                self.client = create_client(self.url, self.service_role_key)
            except Exception:
                self.client = None

    def upload_file(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Uploads bytes to Supabase Storage and returns the object path."""
        if not self.client:
            # Fall back to returning the intended path during scaffold
            return f"{self.bucket}/{path}"
        buf = io.BytesIO(data)
        # Attempt upload; if file exists, set upsert true
        self.client.storage.from_(self.bucket).upload(
            path=path,
            file=buf,
            file_options={"content-type": content_type, "x-upsert": True},
        )
        return f"{self.bucket}/{path}"

    def public_url(self, path: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            res = self.client.storage.from_(self.bucket).get_public_url(path)
            return res
        except Exception:
            return None
