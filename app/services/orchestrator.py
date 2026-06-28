from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.models.job import JobCreateRequest, JobRecord, JobStatus
from app.models.pipeline import (
    CandidateRecord,
    CandidateResult,
    GenerationControlBundle,
    GenerationStrengthControls,
    RegionBlendProfile,
    RegionGatingPolicy,
    PipelineAttempt,
    PipelineDecision,
    PreprocessResult,
    QualityGate,
    ReferenceParseResult,
)
from app.services.artifacts import artifact_service
from app.services.bilingual_summary import bilingual_summary_service
from app.services.generator import (
    ark_mainline_worker,
    generation_router,
    global_generation_worker,
    local_inpaint_worker,
)
from app.services.postprocess import postprocess_service
from app.services.preprocess import preprocess_service
from app.services.reference_parser import reference_parser_service
from app.services.scoring import quality_scoring_service
from app.storage.memory_store import store
from app.utils.images import load_image_bytes


class TaskOrchestrator:
    def create_job(self, payload: JobCreateRequest) -> JobRecord:
        job = JobRecord(**payload.model_dump())
        store.save_job(job)
        return job

    def run_job(self, job_id: str) -> JobRecord:
        job = store.get_job(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        job.status = JobStatus.PREPROCESSING
        store.save_job(job)
        preprocess = preprocess_service.run(job.source_image)
        store.save_preprocess_result(job.job_id, preprocess)

        job.status = JobStatus.PARSING_REFERENCE
        store.save_job(job)
        reference = reference_parser_service.run(job.reference_image)
        store.save_reference_parse_result(job.job_id, reference)

        control_bundle = self._build_control_bundle(job, preprocess)
        store.save_control_bundle(job.job_id, control_bundle)

        job.status = JobStatus.GENERATING
        decision = generation_router.select_pipeline(job, control_bundle)
        store.save_control_bundle(job.job_id, control_bundle)
        job.selected_pipeline = decision.primary_pipeline
        store.save_job(job)

        (
            best_candidate,
            candidate_records,
            postprocessed,
            stage_metadata,
            pipeline_attempts,
            decision,
            control_bundle,
            retry_runs,
            final_rejection_summary,
        ) = self._generate_until_valid(
            job=job,
            preprocess=preprocess,
            control_bundle=control_bundle,
            reference=reference,
            decision=decision,
        )

        if best_candidate is None:
            store.save_candidates(job.job_id, candidate_records)
            job.metadata = {
                "pipeline_decision": decision.model_dump(mode="json"),
                "pipeline_attempts": [attempt.model_dump(mode="json") for attempt in pipeline_attempts],
                "stages": stage_metadata,
                "retry_runs": retry_runs,
                "final_rejection_summary": final_rejection_summary,
                "control_bundle": control_bundle.model_dump(mode="json"),
            }
            job.status = JobStatus.FAILED
            job.failure_code = "NO_VALID_CANDIDATE"
            store.save_job(job)
            return job

        self._persist_outputs(job, postprocessed)
        best_candidate = next((candidate for candidate in postprocessed if candidate.candidate_id == best_candidate.candidate_id), best_candidate)
        best_candidate, candidate_records, final_rejection_summary = self._score_candidates(
            job, postprocessed, preprocess, reference, control_bundle
        )
        store.save_candidates(job.job_id, candidate_records)

        reference_extraction_summary = bilingual_summary_service.build_reference_extraction_summary(
            reference
        )
        transfer_payload_summary = bilingual_summary_service.build_transfer_payload_summary(
            job, reference
        )

        job.result_image = best_candidate.image_url
        job.scores = best_candidate.metadata["scores"]
        job.metadata = {
            "selected_candidate_id": best_candidate.candidate_id,
            "pipeline": best_candidate.pipeline_type,
            "candidate_count": len(postprocessed),
            "generation_provider_mode": best_candidate.metadata.get("provider_mode", "unknown"),
            "local_result_path": best_candidate.metadata.get("local_output_path"),
            "provider_prompt": best_candidate.metadata.get("provider_prompt"),
            "control_bundle": control_bundle.model_dump(mode="json"),
            "reference_extraction_summary": reference_extraction_summary,
            "transfer_payload_summary": transfer_payload_summary,
            "pipeline_decision": decision.model_dump(mode="json"),
            "pipeline_attempts": [attempt.model_dump(mode="json") for attempt in pipeline_attempts],
            "stages": stage_metadata,
            "retry_runs": retry_runs,
            "final_rejection_summary": final_rejection_summary,
        }
        job.status = JobStatus.SUCCEEDED
        store.save_job(job)
        return job

    def _generate_until_valid(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        decision: PipelineDecision,
    ) -> tuple[
        CandidateResult | None,
        list[CandidateRecord],
        list[CandidateResult],
        dict[str, object],
        list[PipelineAttempt],
        PipelineDecision,
        GenerationControlBundle,
        list[dict[str, object]],
        dict[str, int],
    ]:
        gate = control_bundle.quality_gate
        active_bundle = control_bundle
        active_decision = decision
        all_attempts: list[PipelineAttempt] = []
        retry_runs: list[dict[str, object]] = []
        final_stage_metadata: dict[str, object] = {}
        final_postprocessed: list[CandidateResult] = []
        final_records: list[CandidateRecord] = []
        final_best: CandidateResult | None = None
        final_summary: dict[str, int] = {}

        for retry_index in range(gate.max_retry_count + 1):
            if retry_index > 0:
                job.status = JobStatus.RETRYING
                job.retry_count = retry_index
                store.save_job(job)
                store.save_control_bundle(job.job_id, active_bundle)

            candidates, stage_metadata, pipeline_attempts = self._generate_candidates(
                job,
                preprocess,
                active_bundle,
                reference,
                active_decision,
            )
            all_attempts.extend(pipeline_attempts)
            if pipeline_attempts:
                selected_attempt = next(
                    (attempt for attempt in reversed(pipeline_attempts) if attempt.status in {"succeeded", "degraded"}),
                    pipeline_attempts[-1],
                )
                job.selected_pipeline = selected_attempt.pipeline
                store.save_job(job)

            job.status = JobStatus.POSTPROCESSING
            store.save_job(job)
            postprocessed = [
                postprocess_service.run(candidate, preprocess, active_bundle)
                for candidate in candidates
            ]

            job.status = JobStatus.SCORING
            store.save_job(job)
            best_candidate, candidate_records, rejection_summary = self._score_candidates(
                job,
                postprocessed,
                preprocess,
                reference,
                active_bundle,
            )

            final_stage_metadata = stage_metadata
            final_postprocessed = postprocessed
            final_records = candidate_records
            final_best = best_candidate
            final_summary = rejection_summary
            retry_runs.append(
                {
                    "retry_index": retry_index,
                    "decision": active_decision.model_dump(mode="json"),
                    "controls": active_bundle.controls.model_dump(mode="json"),
                    "pipeline_attempt_count": len(pipeline_attempts),
                    "candidate_count": len(postprocessed),
                    "rejection_summary": rejection_summary,
                    "selected_pipeline": job.selected_pipeline,
                    "produced_valid_candidate": best_candidate is not None,
                }
            )

            if best_candidate is not None:
                return (
                    final_best,
                    final_records,
                    final_postprocessed,
                    final_stage_metadata,
                    all_attempts,
                    active_decision,
                    active_bundle,
                    retry_runs,
                    final_summary,
                )

            if retry_index >= gate.max_retry_count:
                break

            active_bundle = self._tune_control_bundle_for_retry(
                active_bundle,
                rejection_summary,
                retry_index + 1,
            )
            active_decision = self._select_retry_decision(job, active_decision, rejection_summary, retry_index + 1)

        return (
            final_best,
            final_records,
            final_postprocessed,
            final_stage_metadata,
            all_attempts,
            active_decision,
            active_bundle,
            retry_runs,
            final_summary,
        )

    def _generate_candidates(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        decision: PipelineDecision,
    ) -> tuple[list[CandidateResult], dict[str, object], list[PipelineAttempt]]:
        stage_metadata: dict[str, object] = {}
        attempts: list[PipelineAttempt] = []

        if decision.primary_pipeline == "ark_native_control_mainline":
            candidates, attempt = ark_mainline_worker.generate(
                job,
                control_bundle,
                reference,
                pipeline_name="ark_native_control_mainline",
            )
            attempts.append(attempt)
            if candidates:
                return candidates, stage_metadata, attempts

            if decision.fallback_pipeline == "ark_hybrid_mainline":
                candidates, stage_metadata, fallback_attempt = self._run_ark_hybrid_mainline(
                    job,
                    preprocess,
                    control_bundle,
                    reference,
                )
                attempts.append(fallback_attempt)
                return candidates, stage_metadata, attempts

            if decision.fallback_pipeline == "two_stage_local_edit":
                candidates, stage_metadata, fallback_attempt = self._run_two_stage_local_edit(
                    job,
                    preprocess,
                    control_bundle,
                    reference,
                )
                attempts.append(fallback_attempt)
                return candidates, stage_metadata, attempts

            return [], stage_metadata, attempts

        if decision.primary_pipeline == "ark_hybrid_mainline":
            candidates, stage_metadata, attempt = self._run_ark_hybrid_mainline(
                job,
                preprocess,
                control_bundle,
                reference,
            )
            attempts.append(attempt)
            if candidates:
                return candidates, stage_metadata, attempts

            if decision.fallback_pipeline == "two_stage_local_edit":
                candidates, stage_metadata, fallback_attempt = self._run_two_stage_local_edit(
                    job,
                    preprocess,
                    control_bundle,
                    reference,
                )
                attempts.append(fallback_attempt)
                return candidates, stage_metadata, attempts

            return [], stage_metadata, attempts

        if decision.primary_pipeline == "global_reference":
            candidates, attempt = global_generation_worker.generate(job, control_bundle, reference)
            attempts.append(attempt)
            return candidates, stage_metadata, attempts

        if decision.primary_pipeline == "two_stage_local_edit":
            candidates, stage_metadata, attempt = self._run_two_stage_local_edit(
                job,
                preprocess,
                control_bundle,
                reference,
            )
            attempts.append(attempt)
            return candidates, stage_metadata, attempts

        candidates, attempt = local_inpaint_worker.generate(job, control_bundle, reference)
        attempts.append(attempt)
        return candidates, stage_metadata, attempts

    def _run_ark_hybrid_mainline(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
    ) -> tuple[list[CandidateResult], dict[str, object], PipelineAttempt]:
        ark_job, oversampling_meta = self._resolve_visual_identity_generation_job(job)
        ark_candidates, ark_attempt = ark_mainline_worker.generate(
            ark_job,
            control_bundle,
            reference,
            pipeline_name="ark_hybrid_mainline",
        )
        if not ark_candidates:
            return [], {
                "mode": "ark_hybrid_mainline",
                "ark_attempt": ark_attempt.model_dump(mode="json"),
                "oversampling": oversampling_meta,
            }, ark_attempt

        self._persist_outputs(job, ark_candidates)

        final_candidates: list[CandidateResult] = []
        refine_attempts: list[dict[str, object]] = []
        runs: list[dict[str, object]] = []
        if job.identity_mode == "visual_identity":
            base_candidates = self._tag_candidates(
                [candidate.model_copy(deep=True) for candidate in ark_candidates],
                pipeline_name="ark_hybrid_mainline",
                stage_name="hybrid_base_stage",
                parent_candidate_id=None,
            )
            for base_index, (base_candidate, ark_candidate) in enumerate(zip(base_candidates, ark_candidates)):
                base_candidate.metadata["provider_mode"] = "ark_http_mainline_hybrid_base"
                base_candidate.metadata["provider_prompt"] = ark_candidate.metadata.get("provider_prompt")
                base_candidate.metadata["local_output_path"] = ark_candidate.metadata.get("local_output_path")
                base_candidate.metadata["oversampling"] = {
                    **oversampling_meta,
                    "candidate_index": base_index,
                }
                base_candidate.metadata["stage_context"] = {
                    **base_candidate.metadata.get("stage_context", {}),
                    "stage_name": "hybrid_base_stage",
                    "visual_identity_direct_pick": True,
                }
            final_candidates.extend(base_candidates)
        for index, ark_candidate in enumerate(ark_candidates):
            source_image = ark_candidate.metadata.get("local_output_path") or ark_candidate.image_url
            makeup_job = job.model_copy(update={"mode": "makeup_only", "candidate_count": 1})
            refine_stage_context = {
                "stage_name": "hybrid_makeup_refine_stage",
                "edit_target": "makeup_only",
                "stage_index": 2,
                "source_image": source_image,
                "parent_candidate_id": ark_candidate.candidate_id,
                "source_stage_candidate_id": ark_candidate.candidate_id,
            }
            refine_stage_context.update(self._build_stage_context_overrides(control_bundle, "hybrid_makeup_refine_stage"))
            refine_candidates, refine_attempt = local_inpaint_worker.generate(
                makeup_job,
                control_bundle,
                reference,
                stage_context=refine_stage_context,
                pipeline_name="ark_hybrid_mainline",
            )
            refine_candidates = self._tag_candidates(
                refine_candidates,
                pipeline_name="ark_hybrid_mainline",
                stage_name="hybrid_makeup_refine_stage",
                parent_candidate_id=ark_candidate.candidate_id,
                sequence_index=index,
            )
            for refine_candidate in refine_candidates:
                refine_candidate.metadata["ark_base_stage"] = {
                    "candidate_id": ark_candidate.candidate_id,
                    "image_url": ark_candidate.image_url,
                    "local_output_path": ark_candidate.metadata.get("local_output_path"),
                    "provider_prompt": ark_candidate.metadata.get("provider_prompt"),
                    "stage_context": ark_candidate.metadata.get("stage_context"),
                }
            final_candidates.extend(refine_candidates)
            refine_attempts.append(refine_attempt.model_dump(mode="json"))
            runs.append(
                {
                    "ark_candidate_id": ark_candidate.candidate_id,
                    "ark_output_path": ark_candidate.metadata.get("local_output_path"),
                    "refine_candidate_ids": [candidate.candidate_id for candidate in refine_candidates],
                    "ark_attempt": ark_attempt.model_dump(mode="json"),
                    "refine_attempt": refine_attempt.model_dump(mode="json"),
                }
            )

        metadata = {
            "mode": "ark_hybrid_mainline",
            "ark_stage_candidate_count": len(ark_candidates),
            "refine_stage_candidate_count": len(final_candidates),
            "visual_identity_base_candidate_count": (
                len(ark_candidates) if job.identity_mode == "visual_identity" else 0
            ),
            "oversampling": oversampling_meta,
            "ark_stage_attempt": ark_attempt.model_dump(mode="json"),
            "refine_stage_attempts": refine_attempts,
            "runs": runs,
        }
        return final_candidates, metadata, PipelineAttempt(
            pipeline="ark_hybrid_mainline",
            status="succeeded" if final_candidates else "failed",
            reason="ark_global_then_local_refine",
            metadata={
                "oversampling": oversampling_meta,
                "ark_stage_attempt": ark_attempt.model_dump(mode="json"),
                "refine_stage_attempts": refine_attempts,
                "run_count": len(runs),
            },
        )

    def _run_two_stage_local_edit(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
    ) -> tuple[list[CandidateResult], dict[str, object], PipelineAttempt]:
        hair_job = job.model_copy(update={"mode": "hair_only"})
        hair_stage_context = {
            "stage_name": "hair_stage",
            "edit_target": "hair_only",
            "stage_index": 1,
            "source_image": control_bundle.source_image,
        }
        hair_stage_context.update(self._build_stage_context_overrides(control_bundle, "hair_stage"))
        hair_candidates, hair_attempt = local_inpaint_worker.generate(
            hair_job,
            control_bundle,
            reference,
            stage_context=hair_stage_context,
            pipeline_name="two_stage_local_edit",
        )
        hair_candidates = self._tag_candidates(
            hair_candidates,
            pipeline_name="two_stage_local_edit",
            stage_name="hair_stage",
            parent_candidate_id=None,
        )
        hair_candidates = [
            postprocess_service.run(candidate, preprocess, control_bundle)
            for candidate in hair_candidates
        ]
        self._persist_outputs(job, hair_candidates)

        final_candidates: list[CandidateResult] = []
        stage_runs: list[dict[str, object]] = []
        makeup_attempts: list[dict[str, object]] = []
        for index, hair_candidate in enumerate(hair_candidates):
            source_image = hair_candidate.metadata.get("local_output_path") or hair_candidate.image_url
            makeup_job = job.model_copy(update={"mode": "makeup_only", "candidate_count": 1})
            makeup_stage_context = {
                "stage_name": "makeup_stage",
                "edit_target": "makeup_only",
                "stage_index": 2,
                "source_image": source_image,
                "parent_candidate_id": hair_candidate.candidate_id,
                "source_stage_candidate_id": hair_candidate.candidate_id,
            }
            makeup_stage_context.update(self._build_stage_context_overrides(control_bundle, "makeup_stage"))
            makeup_candidates, makeup_attempt = local_inpaint_worker.generate(
                makeup_job,
                control_bundle,
                reference,
                stage_context=makeup_stage_context,
                pipeline_name="two_stage_local_edit",
            )
            makeup_candidates = self._tag_candidates(
                makeup_candidates,
                pipeline_name="two_stage_local_edit",
                stage_name="makeup_stage",
                parent_candidate_id=hair_candidate.candidate_id,
                sequence_index=index,
            )
            for makeup_candidate in makeup_candidates:
                makeup_candidate.metadata["hair_stage"] = {
                    "candidate_id": hair_candidate.candidate_id,
                    "image_url": hair_candidate.image_url,
                    "local_output_path": hair_candidate.metadata.get("local_output_path"),
                    "provider_prompt": hair_candidate.metadata.get("provider_prompt"),
                    "stage_context": hair_candidate.metadata.get("stage_context"),
                }
            final_candidates.extend(makeup_candidates)
            makeup_attempts.append(makeup_attempt.model_dump(mode="json"))
            stage_runs.append(
                {
                    "hair_candidate_id": hair_candidate.candidate_id,
                    "hair_output_path": hair_candidate.metadata.get("local_output_path"),
                    "makeup_candidate_ids": [candidate.candidate_id for candidate in makeup_candidates],
                    "hair_attempt": hair_attempt.model_dump(mode="json"),
                    "makeup_attempt": makeup_attempt.model_dump(mode="json"),
                }
            )

        metadata = {
            "mode": "two_stage_local_edit",
            "hair_stage_candidate_count": len(hair_candidates),
            "makeup_stage_candidate_count": len(final_candidates),
            "hair_stage_attempt": hair_attempt.model_dump(mode="json"),
            "makeup_stage_attempts": makeup_attempts,
            "runs": stage_runs,
        }
        return final_candidates, metadata, PipelineAttempt(
            pipeline="two_stage_local_edit",
            status="succeeded" if final_candidates else "failed",
            reason="fallback_inpaint_pipeline",
            metadata={
                "hair_stage_attempt": hair_attempt.model_dump(mode="json"),
                "makeup_stage_attempts": makeup_attempts,
                "run_count": len(stage_runs),
            },
        )

    def _tag_candidates(
        self,
        candidates: list[CandidateResult],
        pipeline_name: str,
        stage_name: str,
        parent_candidate_id: str | None,
        sequence_index: int | None = None,
    ) -> list[CandidateResult]:
        tagged: list[CandidateResult] = []
        for idx, candidate in enumerate(candidates):
            suffix_parts = [stage_name, str(sequence_index if sequence_index is not None else idx)]
            if parent_candidate_id:
                suffix_parts.append(parent_candidate_id)
            candidate.candidate_id = f"{candidate.candidate_id}_{'_'.join(suffix_parts)}"
            candidate.pipeline_type = pipeline_name
            candidate.metadata["stage_name"] = stage_name
            candidate.metadata["parent_candidate_id"] = parent_candidate_id
            candidate.metadata["stage_context"] = {
                **candidate.metadata.get("stage_context", {}),
                "stage_name": stage_name,
                "parent_candidate_id": parent_candidate_id,
            }
            tagged.append(candidate)
        return tagged

    def _score_candidates(
        self,
        job: JobRecord,
        candidates: list[CandidateResult],
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        control_bundle: GenerationControlBundle,
    ) -> tuple[CandidateResult | None, list[CandidateRecord], dict[str, int]]:
        best: CandidateResult | None = None
        best_score = -1.0
        records: list[CandidateRecord] = []
        rejection_summary: dict[str, int] = {}
        gate = control_bundle.quality_gate
        for candidate in candidates:
            scores = quality_scoring_service.score(job, candidate, preprocess, reference)
            candidate.metadata["scores"] = scores
            rejection_reasons = self._evaluate_candidate_rejections(candidate, scores, gate)
            is_valid = not rejection_reasons
            for reason in rejection_reasons:
                rejection_summary[reason] = rejection_summary.get(reason, 0) + 1
            records.append(
                CandidateRecord(
                    candidate_id=candidate.candidate_id,
                    job_id=job.job_id,
                    pipeline_type=candidate.pipeline_type,
                    image_url=candidate.image_url,
                    is_selected=False,
                    scores=scores,
                    metadata={
                        **candidate.metadata,
                        "is_valid": is_valid,
                        "rejection_reasons": rejection_reasons,
                    },
                )
            )
            if rejection_reasons:
                continue
            if scores.final_score > best_score:
                best_score = scores.final_score
                best = candidate
        if best is not None:
            for record in records:
                if record.candidate_id == best.candidate_id:
                    record.is_selected = True
                    break
        return best, records, rejection_summary

    def _evaluate_candidate_rejections(self, candidate: CandidateResult, scores, gate: QualityGate) -> list[str]:
        reasons: list[str] = []
        transfer_metric = candidate.metadata.get("transfer_metric")
        hard_failures: list[str] = []
        if isinstance(transfer_metric, dict):
            for failure in transfer_metric.get("hard_failures", []):
                if isinstance(failure, str) and failure:
                    hard_failures.append(failure)
        visual_identity_override = (
            gate.identity_threshold <= 0.75
            and scores.identity_score >= 0.54
            and scores.transfer_score >= 0.82
            and scores.accessory_score >= 0.95
            and scores.artifact_penalty <= gate.artifact_penalty_threshold
            and not hard_failures
        )
        if scores.identity_score < gate.identity_threshold and not visual_identity_override:
            reasons.append("identity_below_threshold")
        if scores.accessory_score < gate.accessory_threshold:
            reasons.append("accessory_below_threshold")
        if hard_failures:
            reasons.append("transfer_hard_failure")
            reasons.extend([f"transfer_hard_failure:{failure}" for failure in hard_failures])
        if scores.transfer_score < gate.transfer_threshold:
            reasons.append("transfer_below_threshold")
        if scores.artifact_penalty > gate.artifact_penalty_threshold:
            reasons.append("artifact_penalty_above_threshold")
        return reasons

    def _resolve_visual_identity_generation_job(self, job: JobRecord) -> tuple[JobRecord, dict[str, object]]:
        requested_count = int(job.candidate_count)
        effective_count = requested_count
        if job.identity_mode == "visual_identity":
            effective_count = max(requested_count, 4)
        generation_job = job if effective_count == requested_count else job.model_copy(
            update={"candidate_count": effective_count}
        )
        return generation_job, {
            "requested_candidate_count": requested_count,
            "effective_candidate_count": effective_count,
            "applied": effective_count != requested_count,
        }

    def _tune_control_bundle_for_retry(
        self,
        control_bundle: GenerationControlBundle,
        rejection_summary: dict[str, int],
        retry_index: int,
    ) -> GenerationControlBundle:
        controls = control_bundle.controls
        visual_identity_mode = control_bundle.quality_gate.identity_threshold <= 0.75
        identity_failures = rejection_summary.get("identity_below_threshold", 0)
        transfer_failures = rejection_summary.get("transfer_below_threshold", 0)
        accessory_failures = rejection_summary.get("accessory_below_threshold", 0)

        makeup_strength = controls.makeup_strength
        hairstyle_strength = controls.hairstyle_strength
        identity_lock_strength = controls.identity_lock_strength

        if identity_failures > 0:
            if visual_identity_mode:
                makeup_strength = max(0.68, makeup_strength - 0.03)
                hairstyle_strength = max(0.78, hairstyle_strength - 0.025)
                identity_lock_strength = min(0.98, identity_lock_strength + 0.015)
            else:
                makeup_strength = max(0.52, makeup_strength - 0.08)
                hairstyle_strength = max(0.68, hairstyle_strength - 0.06)
                identity_lock_strength = min(1.0, identity_lock_strength + 0.04)
        elif transfer_failures > 0:
            if visual_identity_mode:
                makeup_strength = min(0.95, makeup_strength + 0.05)
                hairstyle_strength = min(0.98, hairstyle_strength + 0.045)
            else:
                makeup_strength = min(0.92, makeup_strength + 0.04)
                hairstyle_strength = min(0.95, hairstyle_strength + 0.04)

        if accessory_failures > 0:
            identity_lock_strength = min(1.0, identity_lock_strength + (0.015 if visual_identity_mode else 0.02))

        return control_bundle.model_copy(
            update={
                "controls": control_bundle.controls.model_copy(
                    update={
                        "makeup_strength": round(makeup_strength, 3),
                        "hairstyle_strength": round(hairstyle_strength, 3),
                        "identity_lock_strength": round(identity_lock_strength, 3),
                    }
                ),
                "pipeline_variant": f"{control_bundle.pipeline_variant}_retry_{retry_index}",
            }
        )

    def _select_retry_decision(
        self,
        job: JobRecord,
        decision: PipelineDecision,
        rejection_summary: dict[str, int],
        retry_index: int,
    ) -> PipelineDecision:
        identity_failures = rejection_summary.get("identity_below_threshold", 0)
        accessory_failures = rejection_summary.get("accessory_below_threshold", 0)
        transfer_failures = rejection_summary.get("transfer_below_threshold", 0)
        if job.identity_mode == "visual_identity":
            if (
                decision.primary_pipeline == "two_stage_local_edit"
                and transfer_failures > 0
                and retry_index >= 1
            ):
                return decision.model_copy(
                    update={
                        "primary_pipeline": "ark_hybrid_mainline",
                        "fallback_pipeline": "two_stage_local_edit",
                        "reason": f"{decision.reason}; retry_route=ark_hybrid_reopen_for_visual_transfer",
                    }
                )
            if (
                decision.primary_pipeline != "two_stage_local_edit"
                and accessory_failures > 0
                and retry_index >= 2
            ):
                return decision.model_copy(
                    update={
                        "primary_pipeline": "two_stage_local_edit",
                        "fallback_pipeline": None,
                        "reason": f"{decision.reason}; retry_route=two_stage_local_edit_for_accessory_guard",
                    }
                )
            return decision
        if decision.primary_pipeline != "two_stage_local_edit" and identity_failures >= max(1, accessory_failures):
            return decision.model_copy(
                update={
                    "primary_pipeline": "two_stage_local_edit",
                    "fallback_pipeline": None,
                    "reason": f"{decision.reason}; retry_route=two_stage_local_edit_for_identity_guard",
                }
            )
        if (
            decision.primary_pipeline == "two_stage_local_edit"
            and transfer_failures > 0
            and identity_failures == 0
            and retry_index >= 2
        ):
            return decision.model_copy(
                update={
                    "primary_pipeline": "ark_hybrid_mainline",
                    "fallback_pipeline": "two_stage_local_edit",
                    "reason": f"{decision.reason}; retry_route=ark_hybrid_reopen_for_transfer",
                }
            )
        return decision

    def _persist_outputs(self, job: JobRecord, candidates: list[CandidateResult]) -> None:
        for candidate in candidates:
            try:
                local_path = artifact_service.persist_candidate_image(candidate, job.job_id)
                candidate.metadata["local_output_path"] = local_path
            except Exception as exc:
                candidate.metadata["local_output_error"] = str(exc)

    def _build_stage_context_overrides(
        self,
        control_bundle: GenerationControlBundle,
        stage_name: str,
    ) -> dict[str, object]:
        policy = control_bundle.region_gating_policy
        if policy is None:
            return {}
        stage_controls = policy.stage_overrides.get(stage_name)
        if stage_controls is None:
            return {
                "region_gating_policy": policy.model_dump(mode="json"),
            }
        return {
            "control_overrides": stage_controls.model_dump(mode="json"),
            "region_gating_policy": policy.model_dump(mode="json"),
        }

    def _build_region_gating_policy(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
    ) -> RegionGatingPolicy:
        visual_identity_mode = job.identity_mode == "visual_identity"
        accessory_ratio = self._mask_fill_ratio(preprocess.accessory_mask.uri)
        hair_ratio = self._mask_fill_ratio(preprocess.editable_hair_mask.uri)
        makeup_ratio = self._mask_fill_ratio(preprocess.editable_makeup_mask.uri)
        feature_ratio = self._mask_fill_ratio(preprocess.feature_lock_mask.uri)
        contour_ratio = self._mask_fill_ratio(preprocess.contour_lock_mask.uri)
        pose_pressure = 0.02 if abs(preprocess.pose.yaw) > 10 or abs(preprocess.pose.pitch) > 10 else 0.0
        accessory_pressure = 0.03 if preprocess.accessory_tags or accessory_ratio >= 0.004 else 0.0

        if visual_identity_mode:
            face_source_weight = self._clamp(
                max(job.identity_lock_strength - 0.08, 0.82) + pose_pressure * 0.7 + accessory_pressure * 0.5,
                0.82,
                0.90,
            )
            feature_source_weight = self._clamp(
                face_source_weight - 0.035 + min(0.02, feature_ratio * 0.16),
                0.78,
                0.88,
            )
            contour_source_weight = self._clamp(
                face_source_weight - 0.085 + min(0.025, contour_ratio * 0.18),
                0.70,
                0.84,
            )
            hair_style_weight = self._clamp(
                0.70 + job.hairstyle_strength * 0.18 + min(0.05, hair_ratio * 0.22),
                0.74,
                0.92,
            )
            makeup_style_weight = self._clamp(
                0.68 + job.makeup_strength * 0.20 + min(0.05, makeup_ratio * 0.24),
                0.72,
                0.90,
            )
        else:
            face_source_weight = self._clamp(
                max(job.identity_lock_strength, 0.88) + pose_pressure + accessory_pressure,
                0.88,
                0.96,
            )
            feature_source_weight = self._clamp(face_source_weight - 0.04 + min(0.02, feature_ratio * 0.18), 0.84, 0.94)
            contour_source_weight = self._clamp(face_source_weight - 0.10 + min(0.03, contour_ratio * 0.22), 0.76, 0.88)
            hair_style_weight = self._clamp(0.62 + job.hairstyle_strength * 0.18 + min(0.05, hair_ratio * 0.22), 0.68, 0.85)
            makeup_style_weight = self._clamp(0.60 + job.makeup_strength * 0.22 + min(0.06, makeup_ratio * 0.28), 0.68, 0.86)

        reasoning = [
            f"identity_mode={job.identity_mode}",
            f"face_source_weight={face_source_weight:.3f}",
            f"feature_source_weight={feature_source_weight:.3f}",
            f"contour_source_weight={contour_source_weight:.3f}",
            f"hair_style_weight={hair_style_weight:.3f}",
            f"makeup_style_weight={makeup_style_weight:.3f}",
            f"accessory_ratio={accessory_ratio:.4f}",
            f"hair_ratio={hair_ratio:.4f}",
            f"makeup_ratio={makeup_ratio:.4f}",
            f"pose=({preprocess.pose.yaw:.2f},{preprocess.pose.pitch:.2f},{preprocess.pose.roll:.2f})",
        ]

        return RegionGatingPolicy(
            strategy="region_adaptive_dual_path_approximation",
            face_core=self._make_profile(
                source_weight=face_source_weight,
                style_weight=1.0 - face_source_weight,
                notes="Preserve face geometry and identity while allowing minor skin-tone adaptation.",
            ),
            feature_lock=self._make_profile(
                source_weight=feature_source_weight,
                style_weight=1.0 - feature_source_weight,
                notes="Brows, eyes, nose bridge, and lip geometry remain source-dominant.",
            ),
            contour=self._make_profile(
                source_weight=contour_source_weight,
                style_weight=1.0 - contour_source_weight,
                notes="Face outline remains source-led to avoid cheek and jaw drift.",
            ),
            accessory=self._make_profile(
                source_weight=1.0 if job.preserve_accessories else 0.85,
                style_weight=0.0 if job.preserve_accessories else 0.15,
                notes="Accessories are hard-preserved whenever requested.",
            ),
            hair=self._make_profile(
                source_weight=1.0 - hair_style_weight,
                style_weight=hair_style_weight,
                notes="Hair region stays style-dominant to maximize hairstyle transfer.",
            ),
            makeup=self._make_profile(
                source_weight=1.0 - makeup_style_weight,
                style_weight=makeup_style_weight,
                notes="Makeup region keeps source facial structure but follows reference cosmetics strongly.",
            ),
            stage_overrides={
                "hair_stage": GenerationStrengthControls(
                    makeup_strength=0.22 if visual_identity_mode else 0.15,
                    hairstyle_strength=round(hair_style_weight, 3),
                    identity_lock_strength=round(feature_source_weight if visual_identity_mode else face_source_weight, 3),
                    preserve_accessories=job.preserve_accessories,
                ),
                "makeup_stage": GenerationStrengthControls(
                    makeup_strength=round(makeup_style_weight, 3),
                    hairstyle_strength=0.28 if visual_identity_mode else 0.2,
                    identity_lock_strength=round(feature_source_weight if visual_identity_mode else face_source_weight, 3),
                    preserve_accessories=job.preserve_accessories,
                ),
                "hybrid_makeup_refine_stage": GenerationStrengthControls(
                    makeup_strength=round(makeup_style_weight, 3),
                    hairstyle_strength=0.24 if visual_identity_mode else 0.18,
                    identity_lock_strength=round(feature_source_weight if visual_identity_mode else face_source_weight, 3),
                    preserve_accessories=job.preserve_accessories,
                ),
            },
            reasoning=reasoning,
        )

    def _build_control_bundle(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
    ) -> GenerationControlBundle:
        region_gating_policy = self._build_region_gating_policy(job, preprocess)
        quality_gate = self._build_quality_gate(job)
        return GenerationControlBundle(
            source_image=job.source_image,
            reference_image=job.reference_image,
            mode=job.mode,
            pipeline_variant=f"{job.identity_mode}_{job.mode}",
            id_mask=preprocess.id_mask,
            style_mask=preprocess.style_mask,
            accessory_mask=preprocess.accessory_mask,
            face_lock_mask=preprocess.face_lock_mask,
            feature_lock_mask=preprocess.feature_lock_mask,
            contour_lock_mask=preprocess.contour_lock_mask,
            editable_hair_mask=preprocess.editable_hair_mask,
            editable_makeup_mask=preprocess.editable_makeup_mask,
            face_bbox=preprocess.face_bbox,
            pose=preprocess.pose,
            landmarks_106=preprocess.landmarks_106,
            face_mesh=preprocess.face_mesh,
            identity_embedding=preprocess.id_embedding,
            controls=GenerationStrengthControls(
                makeup_strength=job.makeup_strength,
                hairstyle_strength=job.hairstyle_strength,
                identity_lock_strength=job.identity_lock_strength,
                preserve_accessories=job.preserve_accessories,
            ),
            region_gating_policy=region_gating_policy,
            quality_gate=quality_gate,
        )

    def _build_quality_gate(self, job: JobRecord) -> QualityGate:
        if job.identity_mode == "visual_identity":
            return QualityGate(
                identity_threshold=0.72,
                accessory_threshold=0.78,
                transfer_threshold=0.74,
                artifact_penalty_threshold=0.22,
                max_retry_count=2,
            )
        return QualityGate()

    def _make_profile(
        self,
        *,
        source_weight: float,
        style_weight: float,
        notes: str,
    ) -> RegionBlendProfile:
        source_weight = round(self._clamp(source_weight, 0.0, 1.0), 3)
        style_weight = round(self._clamp(style_weight, 0.0, 1.0), 3)
        total = source_weight + style_weight
        if total <= 0.0:
            source_weight = 1.0
            style_weight = 0.0
        elif abs(total - 1.0) > 1e-6:
            source_weight = round(source_weight / total, 3)
            style_weight = round(1.0 - source_weight, 3)
        return RegionBlendProfile(
            source_weight=source_weight,
            style_weight=style_weight,
            notes=notes,
        )

    def _mask_fill_ratio(self, mask_ref: str) -> float:
        try:
            mask = Image.open(io.BytesIO(load_image_bytes(mask_ref))).convert("L")
        except Exception:
            return 0.0
        values = np.array(mask, dtype=np.uint8)
        if values.size == 0:
            return 0.0
        return float((values > 0).mean())

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))


orchestrator = TaskOrchestrator()
