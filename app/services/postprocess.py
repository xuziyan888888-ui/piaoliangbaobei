from app.models.pipeline import CandidateResult, PreprocessResult


class PostprocessService:
    def run(self, candidate: CandidateResult, preprocess: PreprocessResult) -> CandidateResult:
        candidate.metadata["postprocessed"] = True
        candidate.metadata["accessory_refill"] = True
        return candidate


postprocess_service = PostprocessService()
