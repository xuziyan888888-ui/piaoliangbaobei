from fastapi import APIRouter, HTTPException

from app.models.job import (
    JobArtifactsResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobDetailResponse,
)
from app.services.orchestrator import orchestrator
from app.storage.memory_store import store

router = APIRouter()


@router.post("/v1/makeup-transfer/jobs", response_model=JobCreateResponse, tags=["jobs"])
def create_job(payload: JobCreateRequest) -> JobCreateResponse:
    job = orchestrator.create_job(payload)
    orchestrator.run_job(job.job_id)
    return JobCreateResponse(job_id=job.job_id, status=job.status)


@router.get("/v1/makeup-transfer/jobs/{job_id}", response_model=JobDetailResponse, tags=["jobs"])
def get_job(job_id: str) -> JobDetailResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobDetailResponse.model_validate(job.model_dump(mode="json"))


@router.get(
    "/v1/makeup-transfer/jobs/{job_id}/artifacts",
    response_model=JobArtifactsResponse,
    tags=["jobs"],
)
def get_job_artifacts(job_id: str) -> JobArtifactsResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    candidates = store.get_candidates(job_id)
    preprocess_result = store.get_preprocess_result(job_id)
    reference_parse_result = store.get_reference_parse_result(job_id)
    control_bundle = store.get_control_bundle(job_id)
    return JobArtifactsResponse(
        job_id=job_id,
        result_image=job.result_image,
        preprocess_result=preprocess_result.model_dump(mode="json") if preprocess_result else None,
        reference_parse_result=(
            reference_parse_result.model_dump(mode="json") if reference_parse_result else None
        ),
        control_bundle=control_bundle.model_dump(mode="json") if control_bundle else None,
        candidates=[candidate.model_dump(mode="json") for candidate in candidates],
    )
