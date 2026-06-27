from __future__ import annotations

from app.models.job import JobCreateRequest, JobRecord, JobStatus
from app.models.pipeline import CandidateRecord, CandidateResult, PipelineAttempt, PipelineDecision
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

        job.status = JobStatus.GENERATING
        decision = generation_router.select_pipeline(job, preprocess, reference)
        job.selected_pipeline = decision.primary_pipeline
        store.save_job(job)

        candidates, stage_metadata, pipeline_attempts = self._generate_candidates(
            job,
            preprocess,
            reference,
            decision,
        )
        if pipeline_attempts:
            selected_attempt = next(
                (attempt for attempt in reversed(pipeline_attempts) if attempt.status in {"succeeded", "degraded"}),
                pipeline_attempts[-1],
            )
            job.selected_pipeline = selected_attempt.pipeline
            store.save_job(job)

        job.status = JobStatus.POSTPROCESSING
        store.save_job(job)
        postprocessed = [postprocess_service.run(candidate, preprocess) for candidate in candidates]

        job.status = JobStatus.SCORING
        store.save_job(job)
        best_candidate, candidate_records = self._score_candidates(
            job, postprocessed, preprocess, reference
        )

        if best_candidate is None:
            store.save_candidates(job.job_id, candidate_records)
            job.metadata = {
                "pipeline_decision": decision.model_dump(mode="json"),
                "pipeline_attempts": [attempt.model_dump(mode="json") for attempt in pipeline_attempts],
                "stages": stage_metadata,
            }
            job.status = JobStatus.FAILED
            job.failure_code = "NO_VALID_CANDIDATE"
            store.save_job(job)
            return job

        self._persist_outputs(job, postprocessed)
        best_candidate, candidate_records = self._score_candidates(
            job, postprocessed, preprocess, reference
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
            "reference_extraction_summary": reference_extraction_summary,
            "transfer_payload_summary": transfer_payload_summary,
            "pipeline_decision": decision.model_dump(mode="json"),
            "pipeline_attempts": [attempt.model_dump(mode="json") for attempt in pipeline_attempts],
            "stages": stage_metadata,
        }
        job.status = JobStatus.SUCCEEDED
        store.save_job(job)
        return job

    def _generate_candidates(
        self,
        job: JobRecord,
        preprocess,
        reference,
        decision: PipelineDecision,
    ) -> tuple[list[CandidateResult], dict[str, object], list[PipelineAttempt]]:
        stage_metadata: dict[str, object] = {}
        attempts: list[PipelineAttempt] = []

        if decision.primary_pipeline == "ark_complete_mainline":
            candidates, attempt = ark_mainline_worker.generate(job, preprocess, reference)
            attempts.append(attempt)
            if candidates:
                return candidates, stage_metadata, attempts

            if decision.fallback_pipeline == "two_stage_local_edit":
                candidates, stage_metadata, fallback_attempt = self._run_two_stage_local_edit(
                    job,
                    preprocess,
                    reference,
                )
                attempts.append(fallback_attempt)
                return candidates, stage_metadata, attempts

            return [], stage_metadata, attempts

        if decision.primary_pipeline == "global_reference":
            candidates, attempt = global_generation_worker.generate(job, preprocess, reference)
            attempts.append(attempt)
            return candidates, stage_metadata, attempts

        if decision.primary_pipeline == "two_stage_local_edit":
            candidates, stage_metadata, attempt = self._run_two_stage_local_edit(
                job,
                preprocess,
                reference,
            )
            attempts.append(attempt)
            return candidates, stage_metadata, attempts

        candidates, attempt = local_inpaint_worker.generate(job, preprocess, reference)
        attempts.append(attempt)
        return candidates, stage_metadata, attempts

    def _run_two_stage_local_edit(
        self,
        job: JobRecord,
        preprocess,
        reference,
    ) -> tuple[list[CandidateResult], dict[str, object], PipelineAttempt]:
        hair_job = job.model_copy(update={"mode": "hair_only"})
        hair_stage_context = {
            "stage_name": "hair_stage",
            "edit_target": "hair_only",
            "stage_index": 1,
            "source_image": job.source_image,
        }
        hair_candidates, hair_attempt = local_inpaint_worker.generate(
            hair_job,
            preprocess,
            reference,
            stage_context=hair_stage_context,
        )
        hair_candidates = self._tag_candidates(
            hair_candidates,
            stage_name="hair_stage",
            parent_candidate_id=None,
        )
        hair_candidates = [postprocess_service.run(candidate, preprocess) for candidate in hair_candidates]
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
            makeup_candidates, makeup_attempt = local_inpaint_worker.generate(
                makeup_job,
                preprocess,
                reference,
                stage_context=makeup_stage_context,
            )
            makeup_candidates = self._tag_candidates(
                makeup_candidates,
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
            candidate.pipeline_type = "two_stage_local_edit"
            candidate.metadata["stage_name"] = stage_name
            candidate.metadata["parent_candidate_id"] = parent_candidate_id
            candidate.metadata["stage_context"] = {
                **candidate.metadata.get("stage_context", {}),
                "stage_name": stage_name,
                "parent_candidate_id": parent_candidate_id,
            }
            tagged.append(candidate)
        return tagged

    def _score_candidates(self, job, candidates, preprocess, reference):
        best: CandidateResult | None = None
        best_score = -1.0
        records: list[CandidateRecord] = []
        for candidate in candidates:
            scores = quality_scoring_service.score(job, candidate, preprocess, reference)
            candidate.metadata["scores"] = scores
            is_valid = scores.identity_score >= 0.9 and scores.accessory_score >= 0.8
            records.append(
                CandidateRecord(
                    candidate_id=candidate.candidate_id,
                    job_id=job.job_id,
                    pipeline_type=candidate.pipeline_type,
                    image_url=candidate.image_url,
                    is_selected=False,
                    scores=scores,
                    metadata={**candidate.metadata, "is_valid": is_valid},
                )
            )
            if scores.identity_score < 0.9:
                continue
            if scores.accessory_score < 0.8:
                continue
            if scores.final_score > best_score:
                best_score = scores.final_score
                best = candidate
        if best is not None:
            for record in records:
                if record.candidate_id == best.candidate_id:
                    record.is_selected = True
                    break
        return best, records

    def _persist_outputs(self, job: JobRecord, candidates: list[CandidateResult]) -> None:
        for candidate in candidates:
            try:
                local_path = artifact_service.persist_candidate_image(candidate, job.job_id)
                candidate.metadata["local_output_path"] = local_path
            except Exception as exc:
                candidate.metadata["local_output_error"] = str(exc)


orchestrator = TaskOrchestrator()
