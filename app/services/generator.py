from urllib.error import HTTPError, URLError

from app.config import settings
from app.models.job import JobRecord
from app.models.pipeline import (
    CandidateResult,
    PipelineAttempt,
    PipelineDecision,
    PreprocessResult,
    ReferenceParseResult,
)
from app.services.model_clients import ArkVisualClient, GenericHTTPImageClient


class GenerationRouter:
    def select_pipeline(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
    ) -> PipelineDecision:
        if job.mode in {"hair_only", "makeup_only"}:
            return PipelineDecision(
                primary_pipeline="local_inpaint",
                reason="single_stage_edit_mode",
                capability_mode="fallback_inpaint",
            )

        if settings.ark_http.mainline_enabled:
            return PipelineDecision(
                primary_pipeline="ark_complete_mainline",
                fallback_pipeline="two_stage_local_edit",
                reason="full_transfer_prefers_ark_mainline",
                capability_mode="mainline_then_fallback",
            )

        return PipelineDecision(
            primary_pipeline="two_stage_local_edit",
            reason="ark_mainline_disabled",
            capability_mode="fallback_inpaint_only",
        )


class ArkMainlineWorker:
    """
    Real Ark mainline binding for the best-known public strong-reference path.

    This worker uses the configured Ark mainline model (for example
    `jimeng_t2i_v40`) as a distinct path from public inpaint. If the call
    fails or the model/action pair is not usable, the orchestrator falls back
    to the two-stage inpaint chain.
    """

    def __init__(self) -> None:
        self._ark_client = ArkVisualClient()

    def generate(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
    ) -> tuple[list[CandidateResult], PipelineAttempt]:
        if not settings.ark_http.mainline_enabled:
            return [], PipelineAttempt(
                pipeline="ark_complete_mainline",
                status="skipped",
                reason="mainline_disabled",
                metadata={"configured": settings.ark_http.mainline_configured},
            )

        if not settings.ark_http.mainline_configured:
            return [], PipelineAttempt(
                pipeline="ark_complete_mainline",
                status="unavailable",
                reason="mainline_model_not_configured",
                metadata={
                    "ark_model": settings.ark_http.model,
                    "generation_route": settings.ark_http.generation_route,
                },
            )

        try:
            candidates = self._ark_client.generate_mainline_candidates(job, preprocess, reference)
            return candidates, PipelineAttempt(
                pipeline="ark_complete_mainline",
                status="succeeded",
                reason="ark_mainline_api",
                metadata={
                    "ark_model": settings.ark_http.model,
                    "generation_route": settings.ark_http.generation_route,
                    "submit_action": self._ark_client._resolve_mainline_submit_action(),
                    "get_action": self._ark_client._resolve_mainline_get_action(),
                },
            )
        except Exception as exc:
            return [], PipelineAttempt(
                pipeline="ark_complete_mainline",
                status="unavailable",
                reason="ark_mainline_error",
                metadata={
                    "ark_model": settings.ark_http.model,
                    "generation_route": settings.ark_http.generation_route,
                    "submit_action": self._ark_client._resolve_mainline_submit_action(),
                    "get_action": self._ark_client._resolve_mainline_get_action(),
                    "error": str(exc),
                },
            )


class LocalInpaintWorker:
    def __init__(self) -> None:
        self._http_client = GenericHTTPImageClient()
        self._ark_client = ArkVisualClient()

    def generate(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        stage_context: dict | None = None,
    ) -> tuple[list[CandidateResult], PipelineAttempt]:
        if settings.local_inpaint.provider == "ark_http" and settings.ark_http.enabled:
            try:
                candidates = self._ark_client.generate_candidates(
                    job, preprocess, reference, stage_context=stage_context
                )
                return candidates, PipelineAttempt(
                    pipeline="local_inpaint",
                    status="succeeded",
                    reason="ark_inpaint",
                    metadata={"provider": "ark_http", "model": settings.local_inpaint.model_name},
                )
            except Exception as exc:
                return (
                    self._mock_generate(
                        job,
                        reason=str(exc),
                        mode="fallback",
                        stage_context=stage_context,
                    ),
                    PipelineAttempt(
                        pipeline="local_inpaint",
                        status="degraded",
                        reason="ark_inpaint_error",
                        metadata={"error": str(exc)},
                    ),
                )

        if settings.local_inpaint.enabled:
            try:
                candidates = self._http_client.generate_candidates(
                    job, preprocess, reference, stage_context=stage_context
                )
                return candidates, PipelineAttempt(
                    pipeline="local_inpaint",
                    status="succeeded",
                    reason="generic_http_inpaint",
                    metadata={"provider": "generic_http", "model": settings.local_inpaint.model_name},
                )
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                return (
                    self._mock_generate(
                        job,
                        reason=str(exc),
                        mode="fallback",
                        stage_context=stage_context,
                    ),
                    PipelineAttempt(
                        pipeline="local_inpaint",
                        status="degraded",
                        reason="generic_http_inpaint_error",
                        metadata={"error": str(exc)},
                    ),
                )

        return (
            self._mock_generate(
                job,
                reason="provider_not_configured",
                mode="mock",
                stage_context=stage_context,
            ),
            PipelineAttempt(
                pipeline="local_inpaint",
                status="degraded",
                reason="provider_not_configured",
                metadata={"provider": settings.local_inpaint.provider},
            ),
        )

    def _mock_generate(
        self,
        job: JobRecord,
        reason: str,
        mode: str,
        stage_context: dict | None = None,
    ) -> list[CandidateResult]:
        stage_context = stage_context or {}
        return [
            CandidateResult(
                candidate_id=f"{job.job_id}_local_{idx}",
                pipeline_type="local_inpaint",
                image_url=f"mock://{job.job_id}/local/{idx}.png",
                metadata={
                    "model": settings.local_inpaint.model_name,
                    "provider_mode": mode,
                    "provider_reason": reason,
                    "stage_context": stage_context,
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
    ) -> tuple[list[CandidateResult], PipelineAttempt]:
        return [
            CandidateResult(
                candidate_id=f"{job.job_id}_global_{idx}",
                pipeline_type="global_reference",
                image_url=f"mock://{job.job_id}/global/{idx}.png",
                metadata={"provider": "ark"},
            )
            for idx in range(job.candidate_count)
        ], PipelineAttempt(
            pipeline="global_reference",
            status="degraded",
            reason="legacy_mock_worker",
        )


generation_router = GenerationRouter()
ark_mainline_worker = ArkMainlineWorker()
local_inpaint_worker = LocalInpaintWorker()
global_generation_worker = GlobalGenerationWorker()
