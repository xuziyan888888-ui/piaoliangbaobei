from __future__ import annotations

import numpy as np

from app.models.job import JobRecord, Scores
from app.models.pipeline import CandidateResult, PreprocessResult, ReferenceParseResult
from app.services.identity import face_identity_service


class QualityScoringService:
    def score(
        self,
        job: JobRecord,
        candidate: CandidateResult,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
    ) -> Scores:
        is_two_stage = candidate.pipeline_type == "two_stage_local_edit"
        stage_name = str(candidate.metadata.get("stage_name") or "")
        is_visual_identity = job.identity_mode == "visual_identity"
        transfer_bonus = 0.03 if is_two_stage else 0.0
        if is_visual_identity and stage_name == "hybrid_base_stage":
            transfer_bonus += 0.04
        identity_score = self._compute_identity_score(job, candidate, preprocess, is_two_stage)
        transfer_score = min(
            0.99,
            0.4 + job.makeup_strength * 0.25 + job.hairstyle_strength * 0.25 + transfer_bonus,
        )
        accessory_score = self._compute_accessory_score(job, candidate, preprocess)
        artifact_penalty = 0.03 if is_two_stage else (0.05 if candidate.pipeline_type == "local_inpaint" else 0.1)
        if is_visual_identity and stage_name == "hybrid_base_stage":
            artifact_penalty = max(0.04, artifact_penalty - 0.02)
        if is_visual_identity:
            final_score = (
                0.35 * identity_score
                + 0.40 * transfer_score
                + 0.15 * accessory_score
                - 0.10 * artifact_penalty
            )
        else:
            final_score = (
                0.45 * identity_score
                + 0.30 * transfer_score
                + 0.15 * accessory_score
                - 0.10 * artifact_penalty
            )
        return Scores(
            identity_score=round(identity_score, 4),
            transfer_score=round(transfer_score, 4),
            accessory_score=round(accessory_score, 4),
            artifact_penalty=round(artifact_penalty, 4),
            final_score=round(final_score, 4),
        )

    def _compute_identity_score(
        self,
        job: JobRecord,
        candidate: CandidateResult,
        preprocess: PreprocessResult,
        is_two_stage: bool,
    ) -> float:
        postprocess_state = candidate.metadata.get("postprocess")
        candidate_ref = candidate.image_url
        if not self._is_postprocessed_candidate(postprocess_state):
            candidate_ref = candidate.metadata.get("local_output_path") or candidate.image_url
        if (
            preprocess.id_embedding.dimension > 0
            and preprocess.id_embedding.vector
            and preprocess.id_embedding.provider != "pseudo_preview"
            and face_identity_service.available
        ):
            candidate_observation = face_identity_service.analyze(str(candidate_ref))
            if candidate_observation is None:
                candidate.metadata["identity_metric"] = {
                    "provider": preprocess.id_embedding.provider,
                    "raw_cosine_similarity": None,
                    "scoring_mode": "candidate_face_not_detected",
                }
                return 0.0

            source_vec = np.asarray(preprocess.id_embedding.vector, dtype=np.float32)
            denom = float(np.linalg.norm(source_vec) * np.linalg.norm(candidate_observation.embedding))
            if denom <= 1e-8:
                candidate.metadata["identity_metric"] = {
                    "provider": preprocess.id_embedding.provider,
                    "raw_cosine_similarity": None,
                    "scoring_mode": "invalid_identity_embedding",
                }
                return 0.0

            similarity = float(np.dot(source_vec, candidate_observation.embedding) / denom)
            similarity = max(-1.0, min(1.0, similarity))
        else:
            similarity = face_identity_service.compare_embedding_to_image(preprocess.id_embedding, candidate_ref)
        if similarity is not None:
            normalized = max(0.0, min(1.0, similarity))
            candidate.metadata["identity_metric"] = {
                "provider": preprocess.id_embedding.provider,
                "raw_cosine_similarity": round(similarity, 4),
                "scoring_mode": "real_face_similarity",
            }
            return normalized

        identity_bonus = 0.02 if is_two_stage else 0.0
        fallback_score = min(0.99, 0.88 + job.identity_lock_strength * 0.1 + identity_bonus)
        candidate.metadata["identity_metric"] = {
            "provider": preprocess.id_embedding.provider,
            "raw_cosine_similarity": None,
            "scoring_mode": "heuristic_fallback",
        }
        return fallback_score

    def _is_postprocessed_candidate(self, postprocess_state: object) -> bool:
        if not isinstance(postprocess_state, dict):
            return False
        status = str(postprocess_state.get("status") or "").lower()
        return status in {"applied", "selected"}

    def _compute_accessory_score(
        self,
        job: JobRecord,
        candidate: CandidateResult,
        preprocess: PreprocessResult,
    ) -> float:
        if not job.preserve_accessories:
            return 0.75

        metric = candidate.metadata.get("accessory_metric")
        if isinstance(metric, dict):
            score = metric.get("score")
            if isinstance(score, (int, float)):
                return max(0.0, min(1.0, float(score)))

        if preprocess.accessory_tags:
            return 0.98
        return 0.9


quality_scoring_service = QualityScoringService()
