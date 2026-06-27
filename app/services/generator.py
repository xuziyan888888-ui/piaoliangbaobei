from urllib.error import HTTPError, URLError

from app.config import settings
from app.models.job import JobRecord
from app.models.pipeline import CandidateResult, PreprocessResult, ReferenceParseResult
from app.services.model_clients import ArkVisualClient, GenericHTTPImageClient


class GenerationRouter:
    def select_pipeline(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
    ) -> str:
        if job.mode in {"hair_only", "makeup_only"}:
            return "local_inpaint"
        if job.preserve_accessories:
            return "local_inpaint"
        return "global_reference"


class LocalInpaintWorker:
    def __init__(self) -> None:
        self._http_client = GenericHTTPImageClient()
        self._ark_client = ArkVisualClient()

    def generate(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
    ) -> list[CandidateResult]:
        if settings.local_inpaint.provider == "ark_http" and settings.ark_http.enabled:
            try:
                return self._ark_client.generate_candidates(job, preprocess, reference)
            except Exception as exc:
                return self._mock_generate(job, reason=str(exc), mode="fallback")

        if settings.local_inpaint.enabled:
            try:
                return self._http_client.generate_candidates(job, preprocess, reference)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                return self._mock_generate(job, reason=str(exc), mode="fallback")

        return self._mock_generate(job, reason="provider_not_configured", mode="mock")

    def _mock_generate(
        self,
        job: JobRecord,
        reason: str,
        mode: str,
    ) -> list[CandidateResult]:
        return [
            CandidateResult(
                candidate_id=f"{job.job_id}_local_{idx}",
                pipeline_type="local_inpaint",
                image_url=f"mock://{job.job_id}/local/{idx}.png",
                metadata={
                    "model": settings.local_inpaint.model_name,
                    "provider_mode": mode,
                    "provider_reason": reason,
                },
            )
            for idx in range(job.candidate_count)
        ]


class GlobalGenerationWorker:
    def generate(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
    ) -> list[CandidateResult]:
        return [
            CandidateResult(
                candidate_id=f"{job.job_id}_global_{idx}",
                pipeline_type="global_reference",
                image_url=f"mock://{job.job_id}/global/{idx}.png",
                metadata={"provider": "ark"},
            )
            for idx in range(job.candidate_count)
        ]


generation_router = GenerationRouter()
local_inpaint_worker = LocalInpaintWorker()
global_generation_worker = GlobalGenerationWorker()
