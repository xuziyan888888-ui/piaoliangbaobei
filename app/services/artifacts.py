import base64
import imghdr
from pathlib import Path
from urllib import request
from urllib.parse import urlparse

from app.models.pipeline import CandidateResult


class ArtifactService:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.outputs_dir = root / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def persist_candidate_image(self, candidate: CandidateResult, job_id: str) -> str:
        suffix = self._guess_suffix(candidate.image_url)
        file_path = self.outputs_dir / f"{candidate.candidate_id}{suffix}"
        self._write_image(candidate.image_url, file_path)
        return str(file_path)

    def _write_image(self, image_ref: str, target_path: Path) -> None:
        if image_ref.startswith("data:image/"):
            _, payload = image_ref.split(",", 1)
            target_path.write_bytes(base64.b64decode(payload))
            return

        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            with request.urlopen(image_ref, timeout=120) as resp:
                target_path.write_bytes(resp.read())
            return

        if image_ref.startswith("mock://"):
            target_path.write_bytes(b"")
            return

        raise ValueError("Unsupported image reference for artifact persistence")

    def _guess_suffix(self, image_ref: str) -> str:
        if image_ref.startswith("data:image/"):
            prefix = image_ref.split(",", 1)[0]
            mime = prefix.replace("data:", "").replace(";base64", "")
            if mime.endswith("png"):
                return ".png"
            if mime.endswith("jpeg") or mime.endswith("jpg"):
                return ".jpg"

        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            path = urlparse(image_ref).path
            suffix = Path(path).suffix
            if suffix and suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                return suffix
            return ".png"

        if image_ref.startswith("mock://"):
            return ".png"

        image_type = imghdr.what(None, h=base64.b64decode(image_ref))
        if image_type == "jpeg":
            return ".jpg"
        if image_type:
            return "." + image_type
        return ".png"


artifact_service = ArtifactService()
