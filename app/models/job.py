from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    CREATED = "created"
    PREPROCESSING = "preprocessing"
    PARSING_REFERENCE = "parsing_reference"
    GENERATING = "generating"
    POSTPROCESSING = "postprocessing"
    SCORING = "scoring"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Scores(BaseModel):
    identity_score: float = 0.0
    transfer_score: float = 0.0
    accessory_score: float = 0.0
    artifact_penalty: float = 0.0
    final_score: float = 0.0


class JobCreateRequest(BaseModel):
    source_image: str = Field(description="Source image URL or base64")
    reference_image: str = Field(description="Reference image URL or base64")
    mode: Literal["full_transfer", "hair_only", "makeup_only"] = "full_transfer"
    identity_mode: Literal["strict_identity", "visual_identity"] = "strict_identity"
    preserve_accessories: bool = True
    makeup_strength: float = Field(default=0.75, ge=0.0, le=1.0)
    hairstyle_strength: float = Field(default=0.85, ge=0.0, le=1.0)
    identity_lock_strength: float = Field(default=0.95, ge=0.0, le=1.0)
    candidate_count: int = Field(default=4, ge=1, le=8)


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: f"job_{uuid4().hex}")
    status: JobStatus = JobStatus.CREATED
    source_image: str
    reference_image: str
    mode: Literal["full_transfer", "hair_only", "makeup_only"] = "full_transfer"
    identity_mode: Literal["strict_identity", "visual_identity"] = "strict_identity"
    preserve_accessories: bool = True
    makeup_strength: float = 0.75
    hairstyle_strength: float = 0.85
    identity_lock_strength: float = 0.95
    candidate_count: int = 4
    selected_pipeline: str | None = None
    retry_count: int = 0
    result_image: str | None = None
    failure_code: str | None = None
    scores: Scores = Field(default_factory=Scores)
    metadata: dict[str, object] = Field(default_factory=dict)


class JobDetailResponse(BaseModel):
    job_id: str
    status: JobStatus
    selected_pipeline: str | None = None
    retry_count: int = 0
    result_image: str | None = None
    failure_code: str | None = None
    scores: Scores
    metadata: dict[str, object]


class JobArtifactsResponse(BaseModel):
    job_id: str
    result_image: str | None = None
    preprocess_result: dict[str, object] | None = None
    reference_parse_result: dict[str, object] | None = None
    control_bundle: dict[str, object] | None = None
    candidates: list[dict[str, object]] = Field(default_factory=list)
