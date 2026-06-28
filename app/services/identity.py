from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

from app.models.pipeline import IdentityEmbeddingAsset
from app.utils.images import load_image_bytes


@dataclass
class FaceIdentityObservation:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    det_score: float
    provider: str
    landmarks_5: np.ndarray | None = None
    landmarks_106: np.ndarray | None = None


class FaceIdentityService:
    def __init__(self) -> None:
        self._face_analysis = None
        self._prepare_attempted = False

    @property
    def available(self) -> bool:
        self._ensure_prepared()
        return self._face_analysis is not None

    def analyze(self, image_ref: str) -> FaceIdentityObservation | None:
        self._ensure_prepared()
        if self._face_analysis is None:
            return None

        image = self._load_bgr_image(image_ref)
        if image is None:
            return None

        try:
            faces = self._face_analysis.get(image)
        except Exception:
            return None
        if not faces:
            return None

        face = max(
            faces,
            key=lambda item: float(
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
        )
        bbox = tuple(int(round(v)) for v in face.bbox[:4])
        embedding = np.asarray(face.embedding, dtype=np.float32)
        kps = np.asarray(getattr(face, "kps", []), dtype=np.float32) if hasattr(face, "kps") else None
        lm106 = (
            np.asarray(getattr(face, "landmark_2d_106", []), dtype=np.float32)
            if hasattr(face, "landmark_2d_106")
            else None
        )
        if kps is not None and kps.size == 0:
            kps = None
        if lm106 is not None and lm106.size == 0:
            lm106 = None

        return FaceIdentityObservation(
            bbox=bbox,
            embedding=embedding,
            det_score=float(getattr(face, "det_score", 0.0)),
            provider="insightface_arcface_buffalo_l",
            landmarks_5=kps,
            landmarks_106=lm106,
        )

    def build_identity_asset(self, image_ref: str) -> IdentityEmbeddingAsset | None:
        observation = self.analyze(image_ref)
        if observation is None:
            return None
        return IdentityEmbeddingAsset(
            vector=[float(v) for v in observation.embedding.tolist()],
            provider=observation.provider,
            dimension=int(observation.embedding.shape[0]),
            source="source_image",
            confidence=round(observation.det_score, 4),
        )

    def compare(self, source_image_ref: str, candidate_image_ref: str) -> float | None:
        source = self.analyze(source_image_ref)
        candidate = self.analyze(candidate_image_ref)
        if source is None or candidate is None:
            return None
        return self._cosine_similarity(source.embedding, candidate.embedding)

    def compare_embedding_to_image(
        self,
        source_embedding: IdentityEmbeddingAsset,
        candidate_image_ref: str,
    ) -> float | None:
        if source_embedding.dimension <= 0 or not source_embedding.vector:
            return None
        candidate = self.analyze(candidate_image_ref)
        if candidate is None:
            return None
        source_vec = np.asarray(source_embedding.vector, dtype=np.float32)
        return self._cosine_similarity(source_vec, candidate.embedding)

    def _ensure_prepared(self) -> None:
        if self._prepare_attempted:
            return
        self._prepare_attempted = True
        if FaceAnalysis is None:
            self._face_analysis = None
            return
        try:
            analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            analyzer.prepare(ctx_id=-1, det_size=(640, 640))
            self._face_analysis = analyzer
        except Exception:
            self._face_analysis = None

    def _load_bgr_image(self, image_ref: str):
        if cv2 is None:
            return None
        try:
            raw = load_image_bytes(image_ref)
        except Exception:
            return None
        try:
            arr = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except Exception:
            pass
        try:
            rgb = Image.open(io.BytesIO(raw)).convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float | None:
        if left.size == 0 or right.size == 0:
            return None
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-8:
            return None
        score = float(np.dot(left, right) / denom)
        return max(-1.0, min(1.0, score))


face_identity_service = FaceIdentityService()
