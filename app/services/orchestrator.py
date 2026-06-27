from app.models.job import JobCreateRequest, JobRecord, JobStatus
from app.models.pipeline import CandidateRecord, CandidateResult
from app.services.artifacts import artifact_service
from app.services.bilingual_summary import bilingual_summary_service
from app.services.generator import (
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
        pipeline = generation_router.select_pipeline(job, preprocess, reference)
        job.selected_pipeline = pipeline
        store.save_job(job)

        if pipeline == "global_reference":
            candidates = global_generation_worker.generate(job, preprocess, reference)
        else:
            candidates = local_inpaint_worker.generate(job, preprocess, reference)

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
        }
        job.status = JobStatus.SUCCEEDED
        store.save_job(job)
        return job

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
                    metadata=candidate.metadata,
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
