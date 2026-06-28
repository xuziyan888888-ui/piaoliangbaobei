from urllib.error import HTTPError, URLError

from app.config import settings
from app.models.job import JobRecord
from app.models.pipeline import (
    CandidateResult,
    GenerationControlBundle,
    MainlineCapabilityProfile,
    PipelineAttempt,
    PipelineDecision,
    ReferenceParseResult,
)
from app.services.model_clients import ArkVisualClient, GenericHTTPImageClient


class ArkCapabilityProbe:
    def probe(self) -> MainlineCapabilityProfile:
        if not settings.ark_http.mainline_enabled:
            return MainlineCapabilityProfile(
                mainline_mode="disabled",
                supports_reference_image=False,
                supports_multi_image_reference=False,
                evidence_level="official_confirmed",
                confirmed_surfaces=[],
                missing_surfaces=[
                    "mainline_disabled",
                    "multi_image_reference",
                    "executable_masks",
                    "identity_embedding",
                ],
                control_surface="disabled",
                summary="Ark mainline is disabled in config.",
            )

        if not settings.ark_http.mainline_configured:
            return MainlineCapabilityProfile(
                mainline_mode="disabled",
                supports_reference_image=False,
                supports_multi_image_reference=False,
                evidence_level="official_confirmed",
                confirmed_surfaces=[],
                missing_surfaces=[
                    "mainline_model_not_configured",
                    "multi_image_reference",
                    "executable_masks",
                    "identity_embedding",
                ],
                control_surface="disabled",
                summary="Ark mainline is enabled but no mainline model is configured.",
            )

        # P5 conclusion:
        # Official public materials confirm:
        # - multi-image/reference-image generation surfaces
        # - inpainting/local edit surfaces with mask image transport
        # Official public materials do not confirm:
        # - executable identity embedding inputs
        # - native hard mask/control-image identity locking for the mainline API
        official_multi_image_reference = settings.ark_http.supports_multi_image_reference
        official_inpaint_mask_surface = True
        official_executable_masks = False
        official_control_image = False
        official_identity_embedding = False

        confirmed_surfaces = ["public_multi_image_reference", "public_inpaint_mask_transport"]
        missing_surfaces = ["native_identity_embedding", "native_mainline_executable_mask_control"]
        evidence_level = "official_confirmed"

        supports_executable_masks = official_executable_masks
        supports_control_image = official_control_image
        supports_identity_embedding = official_identity_embedding
        supports_multi_image_reference = official_multi_image_reference

        if supports_executable_masks and (supports_control_image or supports_identity_embedding):
            return MainlineCapabilityProfile(
                mainline_mode="native_executable",
                supports_executable_masks=supports_executable_masks,
                supports_control_image=supports_control_image,
                supports_identity_embedding=supports_identity_embedding,
                supports_reference_image=True,
                supports_multi_image_reference=supports_multi_image_reference,
                evidence_level=evidence_level,
                confirmed_surfaces=confirmed_surfaces,
                missing_surfaces=missing_surfaces,
                control_surface="native_controls",
                summary="Ark mainline is configured with native executable control support.",
            )

        return MainlineCapabilityProfile(
            mainline_mode="hybrid",
            supports_executable_masks=supports_executable_masks,
            supports_control_image=supports_control_image,
            supports_identity_embedding=supports_identity_embedding,
            supports_reference_image=True,
            supports_multi_image_reference=supports_multi_image_reference,
            evidence_level=evidence_level,
            confirmed_surfaces=confirmed_surfaces,
            missing_surfaces=missing_surfaces,
            control_surface="hybrid_controls",
            summary="Officially confirmed public Ark surfaces cover multi-image reference and inpaint mask transport, but not native executable identity/mask control for the mainline API; treat it as a hybrid base generation stage.",
        )


class GenerationRouter:
    def __init__(self) -> None:
        self._probe = ArkCapabilityProbe()

    def select_pipeline(
        self,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
    ) -> PipelineDecision:
        if job.mode in {"hair_only", "makeup_only"}:
            control_bundle.pipeline_variant = "local_inpaint"
            control_bundle.delivery_mode = "fallback_only"
            return PipelineDecision(
                primary_pipeline="local_inpaint",
                reason="single_stage_edit_mode",
                capability_mode="fallback_inpaint",
            )

        capability_profile = self._probe.probe()
        control_bundle.capability_profile = capability_profile

        if capability_profile.mainline_mode == "native_executable":
            control_bundle.pipeline_variant = "ark_native_control_mainline"
            control_bundle.delivery_mode = "native_controls"
            return PipelineDecision(
                primary_pipeline="ark_native_control_mainline",
                fallback_pipeline="ark_hybrid_mainline",
                reason="ark_native_controls_available",
                capability_mode="native_mainline",
                capability_profile=capability_profile,
            )

        if capability_profile.mainline_mode == "hybrid":
            control_bundle.pipeline_variant = "ark_hybrid_mainline"
            control_bundle.delivery_mode = "hybrid_controls"
            return PipelineDecision(
                primary_pipeline="ark_hybrid_mainline",
                fallback_pipeline="two_stage_local_edit",
                reason="ark_requires_hybrid_control_closure",
                capability_mode="hybrid_mainline",
                capability_profile=capability_profile,
            )

        control_bundle.pipeline_variant = "two_stage_local_edit"
        control_bundle.delivery_mode = "fallback_only"
        return PipelineDecision(
            primary_pipeline="two_stage_local_edit",
            reason="ark_mainline_unavailable",
            capability_mode="fallback_inpaint_only",
            capability_profile=capability_profile,
        )


class ArkMainlineWorker:
    """
    Ark mainline worker used by both native-control and hybrid-control modes.

    The worker always binds to the configured Ark mainline model, while the
    orchestration layer decides whether the generated result is final
    (`ark_native_control_mainline`) or only the first stage of a hybrid flow
    (`ark_hybrid_mainline`).
    """

    def __init__(self) -> None:
        self._ark_client = ArkVisualClient()

    def generate(
        self,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        pipeline_name: str,
    ) -> tuple[list[CandidateResult], PipelineAttempt]:
        if not settings.ark_http.mainline_enabled:
            return [], PipelineAttempt(
                pipeline=pipeline_name,
                status="skipped",
                reason="mainline_disabled",
                metadata={"configured": settings.ark_http.mainline_configured},
            )

        if not settings.ark_http.mainline_configured:
            return [], PipelineAttempt(
                pipeline=pipeline_name,
                status="unavailable",
                reason="mainline_model_not_configured",
                metadata={
                    "ark_model": settings.ark_http.model,
                    "generation_route": settings.ark_http.generation_route,
                },
            )

        try:
            candidates = self._ark_client.generate_mainline_candidates(
                job,
                control_bundle,
                reference,
                pipeline_type=pipeline_name,
            )
            return candidates, PipelineAttempt(
                pipeline=pipeline_name,
                status="succeeded",
                reason="ark_mainline_api",
                metadata={
                    "ark_model": settings.ark_http.model,
                    "generation_route": settings.ark_http.generation_route,
                    "submit_action": self._ark_client._resolve_mainline_submit_action(),
                    "get_action": self._ark_client._resolve_mainline_get_action(),
                    "capability_profile": (
                        control_bundle.capability_profile.model_dump(mode="json")
                        if control_bundle.capability_profile
                        else None
                    ),
                },
            )
        except Exception as exc:
            return [], PipelineAttempt(
                pipeline=pipeline_name,
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
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        stage_context: dict | None = None,
        pipeline_name: str = "local_inpaint",
    ) -> tuple[list[CandidateResult], PipelineAttempt]:
        if settings.local_inpaint.provider == "ark_http" and settings.ark_http.enabled:
            try:
                candidates = self._ark_client.generate_candidates(
                    job,
                    control_bundle,
                    reference,
                    stage_context=stage_context,
                    pipeline_type=pipeline_name,
                )
                return candidates, PipelineAttempt(
                    pipeline=pipeline_name,
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
                        pipeline_name=pipeline_name,
                    ),
                    PipelineAttempt(
                        pipeline=pipeline_name,
                        status="degraded",
                        reason="ark_inpaint_error",
                        metadata={"error": str(exc)},
                    ),
                )

        if settings.local_inpaint.enabled:
            try:
                candidates = self._http_client.generate_candidates(
                    job,
                    control_bundle,
                    reference,
                    stage_context=stage_context,
                    pipeline_type=pipeline_name,
                )
                return candidates, PipelineAttempt(
                    pipeline=pipeline_name,
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
                        pipeline_name=pipeline_name,
                    ),
                    PipelineAttempt(
                        pipeline=pipeline_name,
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
                pipeline_name=pipeline_name,
            ),
            PipelineAttempt(
                pipeline=pipeline_name,
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
        pipeline_name: str = "local_inpaint",
    ) -> list[CandidateResult]:
        stage_context = stage_context or {}
        return [
            CandidateResult(
                candidate_id=f"{job.job_id}_local_{idx}",
                pipeline_type=pipeline_name,
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
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
    ) -> tuple[list[CandidateResult], PipelineAttempt]:
        return [
            CandidateResult(
                candidate_id=f"{job.job_id}_global_{idx}",
                pipeline_type="global_reference",
                image_url=f"mock://{job.job_id}/global/{idx}.png",
                metadata={
                    "provider": "ark",
                    "pipeline_variant": control_bundle.pipeline_variant,
                    "reference_caption": reference.style_caption,
                },
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
