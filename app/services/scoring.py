from __future__ import annotations

import numpy as np

from app.models.job import JobRecord, Scores
from app.models.pipeline import (
    BangsFeatures,
    CandidateResult,
    HairFeatures,
    MakeupFeatures,
    PreprocessResult,
    ReferenceParseResult,
)
from app.services.identity import face_identity_service
from app.services.reference_parser import reference_parser_service


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
        transfer_score = self._compute_transfer_score(job, candidate, reference, transfer_bonus)
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

    def _compute_transfer_score(
        self,
        job: JobRecord,
        candidate: CandidateResult,
        reference: ReferenceParseResult,
        transfer_bonus: float,
    ) -> float:
        cached_metric = candidate.metadata.get("transfer_metric")
        if isinstance(cached_metric, dict):
            cached_score = cached_metric.get("score")
            if isinstance(cached_score, (int, float)):
                return max(0.0, min(0.99, float(cached_score)))

        fallback_score = min(
            0.99,
            0.4 + job.makeup_strength * 0.25 + job.hairstyle_strength * 0.25 + transfer_bonus,
        )

        candidate_ref = self._resolve_candidate_reference(candidate)
        try:
            candidate_parse = reference_parser_service.run(str(candidate_ref))
        except Exception as exc:
            candidate.metadata["transfer_metric"] = {
                "scoring_mode": "heuristic_fallback_parse_error",
                "score": round(fallback_score, 4),
                "error": str(exc),
                "hard_failures": [],
            }
            return fallback_score

        hair_metric = self._compare_hair(reference.hair_features, reference.bangs, candidate_parse.hair_features, candidate_parse.bangs)
        makeup_metric = self._compare_makeup(reference.makeup_features, candidate_parse.makeup_features)
        hard_failures = [*hair_metric["hard_failures"], *makeup_metric["hard_failures"]]

        raw_score = 0.58 * float(hair_metric["score"]) + 0.42 * float(makeup_metric["score"]) + transfer_bonus
        transfer_score = self._clamp(raw_score, 0.0, 0.99)
        if hard_failures:
            hard_cap = max(0.18, 0.58 - 0.12 * (len(hard_failures) - 1))
            transfer_score = min(transfer_score, hard_cap)

        candidate.metadata["transfer_metric"] = {
            "scoring_mode": "image_reparse_comparison",
            "score": round(transfer_score, 4),
            "raw_score_before_fail_cap": round(raw_score, 4),
            "hair_score": round(float(hair_metric["score"]), 4),
            "makeup_score": round(float(makeup_metric["score"]), 4),
            "hard_failures": hard_failures,
            "hard_failure_count": len(hard_failures),
            "hair_components": hair_metric["components"],
            "makeup_components": makeup_metric["components"],
            "reference_signature": self._build_reference_signature(reference),
            "candidate_signature": self._build_reference_signature(candidate_parse),
        }
        return transfer_score

    def _compare_hair(
        self,
        reference_hair: HairFeatures,
        reference_bangs: BangsFeatures,
        candidate_hair: HairFeatures,
        candidate_bangs: BangsFeatures,
    ) -> dict[str, object]:
        hard_failures: list[str] = []

        reference_requires_updo = self._is_updo(reference_hair)
        candidate_is_updo = self._is_updo(candidate_hair)
        if reference_requires_updo and not candidate_is_updo:
            hard_failures.append("updo_missing")

        reference_requires_bangs = bool(reference_bangs.exists)
        candidate_has_bangs = bool(candidate_bangs.exists)
        if reference_requires_bangs and not candidate_has_bangs:
            hard_failures.append("bangs_missing")

        reference_requires_side_locks = bool(reference_hair.side_locks.exists and reference_hair.side_locks.intensity >= 0.4)
        candidate_has_side_locks = bool(candidate_hair.side_locks.exists and candidate_hair.side_locks.intensity >= 0.2)
        if reference_requires_side_locks and not candidate_has_side_locks:
            hard_failures.append("side_locks_missing")

        if reference_requires_updo:
            if candidate_is_updo:
                updo_score = 1.0 if candidate_hair.updo_type == reference_hair.updo_type else 0.78
            else:
                updo_score = 0.0
        else:
            updo_score = 1.0 if candidate_is_updo == reference_requires_updo else 0.35

        if reference_requires_bangs:
            if candidate_has_bangs:
                bangs_score = 1.0 if candidate_bangs.type == reference_bangs.type else 0.8
                bangs_score = 0.65 * bangs_score + 0.35 * self._float_similarity(
                    reference_bangs.density,
                    candidate_bangs.density,
                    tolerance=0.28,
                )
            else:
                bangs_score = 0.0
        else:
            bangs_score = 1.0 if not candidate_has_bangs else 0.4

        if reference_requires_side_locks:
            if candidate_has_side_locks:
                side_locks_score = 0.55 + 0.45 * self._float_similarity(
                    reference_hair.side_locks.intensity,
                    candidate_hair.side_locks.intensity,
                    tolerance=0.24,
                )
            else:
                side_locks_score = 0.0
        else:
            side_locks_score = 1.0 if not candidate_has_side_locks else 0.55

        parting_score = self._parting_similarity(reference_hair.parting, candidate_hair.parting)
        crown_score = self._float_similarity(reference_hair.volume_crown, candidate_hair.volume_crown, tolerance=0.22)
        hairline_score = self._float_similarity(
            reference_hair.hairline_exposure,
            candidate_hair.hairline_exposure,
            tolerance=0.22,
        )

        components = {
            "updo": round(updo_score, 4),
            "bangs": round(bangs_score, 4),
            "side_locks": round(side_locks_score, 4),
            "parting": round(parting_score, 4),
            "crown_volume": round(crown_score, 4),
            "hairline_exposure": round(hairline_score, 4),
        }
        score = (
            0.32 * updo_score
            + 0.18 * bangs_score
            + 0.14 * side_locks_score
            + 0.12 * parting_score
            + 0.12 * crown_score
            + 0.12 * hairline_score
        )
        return {
            "score": self._clamp(score, 0.0, 1.0),
            "components": components,
            "hard_failures": hard_failures,
        }

    def _compare_makeup(
        self,
        reference_makeup: MakeupFeatures,
        candidate_makeup: MakeupFeatures,
    ) -> dict[str, object]:
        hard_failures: list[str] = []

        lip_color_score = self._color_similarity(
            reference_makeup.lips.color,
            candidate_makeup.lips.color,
        )
        lip_intensity_score = self._float_similarity(
            reference_makeup.lips.intensity,
            candidate_makeup.lips.intensity,
            tolerance=0.24,
        )
        lips_score = 0.55 * lip_color_score + 0.45 * lip_intensity_score

        blush_color_score = self._color_similarity(
            reference_makeup.blush.color,
            candidate_makeup.blush.color,
        )
        blush_intensity_score = self._float_similarity(
            reference_makeup.blush.intensity,
            candidate_makeup.blush.intensity,
            tolerance=0.22,
        )
        blush_score = 0.5 * blush_color_score + 0.5 * blush_intensity_score

        eyeliner_style_score = self._categorical_similarity(
            reference_makeup.eyeliner.style,
            candidate_makeup.eyeliner.style,
        )
        eyeliner_color_score = self._color_similarity(
            reference_makeup.eyeliner.color,
            candidate_makeup.eyeliner.color,
        )
        eyeliner_intensity_score = self._float_similarity(
            reference_makeup.eyeliner.intensity,
            candidate_makeup.eyeliner.intensity,
            tolerance=0.22,
        )
        eyeliner_score = (
            0.35 * eyeliner_style_score
            + 0.25 * eyeliner_color_score
            + 0.40 * eyeliner_intensity_score
        )

        eyeshadow_color_score = self._color_similarity(
            reference_makeup.eyeshadow.main_color,
            candidate_makeup.eyeshadow.main_color,
        )
        eyeshadow_secondary_score = self._color_similarity(
            reference_makeup.eyeshadow.secondary_color,
            candidate_makeup.eyeshadow.secondary_color,
        )
        eyeshadow_intensity_score = self._float_similarity(
            reference_makeup.eyeshadow.intensity,
            candidate_makeup.eyeshadow.intensity,
            tolerance=0.22,
        )
        eyeshadow_score = (
            0.32 * eyeshadow_color_score
            + 0.18 * eyeshadow_secondary_score
            + 0.50 * eyeshadow_intensity_score
        )

        base_finish_score = self._categorical_similarity(
            reference_makeup.base_makeup.finish,
            candidate_makeup.base_makeup.finish,
        )
        base_intensity_score = self._float_similarity(
            reference_makeup.base_makeup.intensity,
            candidate_makeup.base_makeup.intensity,
            tolerance=0.2,
        )
        base_score = 0.45 * base_finish_score + 0.55 * base_intensity_score

        if reference_makeup.eyeshadow.intensity >= 0.3 and candidate_makeup.eyeshadow.intensity <= reference_makeup.eyeshadow.intensity * 0.45:
            hard_failures.append("eyeshadow_too_weak")
        if reference_makeup.blush.intensity >= 0.26 and candidate_makeup.blush.intensity <= reference_makeup.blush.intensity * 0.45:
            hard_failures.append("blush_too_weak")
        if reference_makeup.lips.intensity >= 0.2 and candidate_makeup.lips.intensity <= reference_makeup.lips.intensity * 0.45:
            hard_failures.append("lip_transfer_too_weak")

        components = {
            "lips": round(lips_score, 4),
            "blush": round(blush_score, 4),
            "eyeliner": round(eyeliner_score, 4),
            "eyeshadow": round(eyeshadow_score, 4),
            "base": round(base_score, 4),
        }
        score = (
            0.28 * lips_score
            + 0.20 * blush_score
            + 0.18 * eyeliner_score
            + 0.24 * eyeshadow_score
            + 0.10 * base_score
        )
        return {
            "score": self._clamp(score, 0.0, 1.0),
            "components": components,
            "hard_failures": hard_failures,
        }

    def _resolve_candidate_reference(self, candidate: CandidateResult) -> str:
        postprocess_state = candidate.metadata.get("postprocess")
        if self._is_postprocessed_candidate(postprocess_state):
            return str(candidate.image_url)
        return str(candidate.metadata.get("local_output_path") or candidate.image_url)

    def _build_reference_signature(self, reference: ReferenceParseResult) -> dict[str, object]:
        return {
            "hair": {
                "primary_style": reference.hair_features.primary_style,
                "updo_type": reference.hair_features.updo_type,
                "parting": reference.hair_features.parting,
                "volume_crown": round(reference.hair_features.volume_crown, 4),
                "hairline_exposure": round(reference.hair_features.hairline_exposure, 4),
                "side_locks_exists": reference.hair_features.side_locks.exists,
                "side_locks_intensity": round(reference.hair_features.side_locks.intensity, 4),
            },
            "bangs": {
                "exists": reference.bangs.exists,
                "type": reference.bangs.type,
                "density": round(reference.bangs.density, 4),
            },
            "makeup": {
                "lips_color": reference.makeup_features.lips.color,
                "lips_intensity": round(reference.makeup_features.lips.intensity, 4),
                "blush_color": reference.makeup_features.blush.color,
                "blush_intensity": round(reference.makeup_features.blush.intensity, 4),
                "eyeliner_style": reference.makeup_features.eyeliner.style,
                "eyeliner_intensity": round(reference.makeup_features.eyeliner.intensity, 4),
                "eyeshadow_main_color": reference.makeup_features.eyeshadow.main_color,
                "eyeshadow_intensity": round(reference.makeup_features.eyeshadow.intensity, 4),
                "base_finish": reference.makeup_features.base_makeup.finish,
            },
        }

    def _is_updo(self, hair: HairFeatures) -> bool:
        return hair.style == "updo" or "updo" in hair.primary_style

    def _parting_similarity(self, reference_parting: str, candidate_parting: str) -> float:
        if reference_parting == candidate_parting:
            return 1.0
        if reference_parting == "middle":
            return 0.25 if candidate_parting.startswith("side_") else 0.45
        if reference_parting.startswith("side_") and candidate_parting.startswith("side_"):
            return 0.75 if reference_parting.split("_")[1][0] == candidate_parting.split("_")[1][0] else 0.45
        if reference_parting in {"none_or_natural_back", "unknown"} or candidate_parting in {"none_or_natural_back", "unknown"}:
            return 0.55
        return 0.3

    def _color_similarity(self, reference_color: str, candidate_color: str) -> float:
        if reference_color == candidate_color:
            return 1.0
        if reference_color in {"unknown", "none"} or candidate_color in {"unknown", "none"}:
            return 0.45
        if self._color_family(reference_color) == self._color_family(candidate_color):
            return 0.74
        return 0.28

    def _color_family(self, color: str) -> str:
        normalized = color.lower()
        if any(token in normalized for token in ("beige", "ivory", "champagne")):
            return "light_neutral"
        if any(token in normalized for token in ("peach", "coral", "rose", "pink")):
            return "warm_soft"
        if "brown" in normalized:
            return "brown"
        if "taupe" in normalized or "mauve" in normalized:
            return "cool_neutral"
        if "charcoal" in normalized or "black" in normalized:
            return "deep"
        return normalized

    def _categorical_similarity(self, reference_value: str, candidate_value: str) -> float:
        if reference_value == candidate_value:
            return 1.0
        if reference_value in {"unknown", "none"} or candidate_value in {"unknown", "none"}:
            return 0.45
        if reference_value.split("_")[0] == candidate_value.split("_")[0]:
            return 0.72
        return 0.3

    def _float_similarity(self, reference_value: float, candidate_value: float, tolerance: float) -> float:
        if tolerance <= 0:
            return 1.0 if abs(reference_value - candidate_value) <= 1e-6 else 0.0
        distance = abs(float(reference_value) - float(candidate_value))
        return self._clamp(1.0 - distance / tolerance, 0.0, 1.0)

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

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
