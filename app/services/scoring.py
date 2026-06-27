from app.models.job import JobRecord, Scores
from app.models.pipeline import CandidateResult, PreprocessResult, ReferenceParseResult


class QualityScoringService:
    def score(
        self,
        job: JobRecord,
        candidate: CandidateResult,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
    ) -> Scores:
        identity_score = min(0.99, 0.88 + job.identity_lock_strength * 0.1)
        transfer_score = min(
            0.99,
            0.4 + job.makeup_strength * 0.25 + job.hairstyle_strength * 0.25,
        )
        accessory_score = 0.98 if job.preserve_accessories else 0.75
        artifact_penalty = 0.05 if candidate.pipeline_type == "local_inpaint" else 0.1
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


quality_scoring_service = QualityScoringService()
