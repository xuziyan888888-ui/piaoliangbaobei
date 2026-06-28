from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from app.models.pipeline import CandidateResult, GenerationControlBundle, PreprocessResult
from app.services.identity import face_identity_service
from app.utils.images import load_image_bytes


class PostprocessService:
    _MODE_PRIORITY = {
        "none": 0,
        "accessory_only": 1,
        "light_identity": 2,
        "full_identity": 3,
        "region_adaptive": 4,
    }

    def run(
        self,
        candidate: CandidateResult,
        preprocess: PreprocessResult,
        control_bundle: GenerationControlBundle | None = None,
    ) -> CandidateResult:
        source_ref = preprocess.source_image_ref
        candidate_ref = candidate.metadata.get("local_output_path") or candidate.image_url
        if not source_ref:
            candidate.metadata["postprocessed"] = False
            candidate.metadata["postprocess"] = {
                "status": "skipped",
                "reason": "missing_source_image_ref",
            }
            return candidate

        try:
            source_image = self._load_rgb(source_ref)
            generated_image = self._load_rgb(str(candidate_ref))
        except Exception as exc:
            candidate.metadata["postprocessed"] = False
            candidate.metadata["postprocess"] = {
                "status": "skipped",
                "reason": "image_load_failed",
                "error": str(exc),
            }
            return candidate

        resized_to_source = False
        if generated_image.size != source_image.size:
            generated_image = generated_image.resize(source_image.size, Image.Resampling.LANCZOS)
            resized_to_source = True

        try:
            accessory_mask = self._load_mask(preprocess.accessory_mask.uri, source_image.size)
            face_lock_mask = self._load_mask(preprocess.face_lock_mask.uri, source_image.size)
            feature_lock_mask = self._load_mask(preprocess.feature_lock_mask.uri, source_image.size)
            contour_lock_mask = self._load_mask(preprocess.contour_lock_mask.uri, source_image.size)
            makeup_mask = self._load_mask(preprocess.editable_makeup_mask.uri, source_image.size)
            hair_mask = self._load_mask(preprocess.editable_hair_mask.uri, source_image.size)
        except Exception as exc:
            candidate.metadata["postprocessed"] = False
            candidate.metadata["postprocess"] = {
                "status": "skipped",
                "reason": "mask_load_failed",
                "error": str(exc),
            }
            return candidate

        original_generated = generated_image.copy()
        visual_identity_mode = bool(
            control_bundle
            and control_bundle.quality_gate.identity_threshold <= 0.75
        )
        core_face_mask = self._build_core_face_protect_mask(
            face_lock_mask=face_lock_mask,
            makeup_mask=makeup_mask,
            accessory_mask=accessory_mask,
        )
        identity_hard_mask = self._build_identity_hard_protect_mask(
            face_lock_mask=face_lock_mask,
            feature_lock_mask=feature_lock_mask,
            contour_lock_mask=contour_lock_mask,
            makeup_mask=makeup_mask,
            hair_mask=hair_mask,
            accessory_mask=accessory_mask,
        )
        hard_accessory_mask = self._dilate_mask(
            accessory_mask,
            radius=self._scaled_radius(source_image.size, minimum=1, maximum=4, fraction=0.0025),
        )
        hard_visual_accessory_mask = self._dilate_mask(
            self._build_visual_accessory_protect_mask(
                accessory_mask=accessory_mask,
                face_lock_mask=face_lock_mask,
                feature_lock_mask=feature_lock_mask,
            ),
            radius=self._scaled_radius(source_image.size, minimum=1, maximum=4, fraction=0.0025),
        )
        preserve_accessory_mask = hard_visual_accessory_mask if visual_identity_mode else hard_accessory_mask

        mode_reports: dict[str, dict[str, object]] = {}
        mode_payloads: dict[str, str] = {}
        selection_basis = (
            "arcface_identity"
            if self._can_score_identity(preprocess)
            else "heuristic_accessory_and_change_ratio"
        )
        region_policy = control_bundle.region_gating_policy if control_bundle else None
        shared_context = {
            "core_face_mask": core_face_mask,
            "identity_hard_mask": identity_hard_mask,
            "feature_lock_mask": feature_lock_mask,
            "contour_lock_mask": contour_lock_mask,
            "accessory_mask": accessory_mask,
            "hard_accessory_mask": hard_accessory_mask,
            "hard_visual_accessory_mask": hard_visual_accessory_mask,
            "preserve_accessory_mask": preserve_accessory_mask,
            "makeup_mask": makeup_mask,
            "hair_mask": hair_mask,
            "region_policy": region_policy,
            "visual_identity_mode": visual_identity_mode,
        }

        for mode in self._MODE_PRIORITY:
            blended, metrics = self._apply_mode(
                mode=mode,
                source_image=source_image,
                generated_image=generated_image,
                shared_context=shared_context,
            )
            metrics["changed_ratio"] = round(self._changed_ratio(original_generated, blended), 4)
            accessory_similarity = self._masked_similarity(source_image, blended, preserve_accessory_mask)
            if accessory_similarity is not None:
                metrics["accessory_similarity_after"] = round(accessory_similarity, 4)

            payload = self._to_data_url(blended)
            mode_payloads[mode] = payload
            identity_score = self._compute_identity_score(preprocess, payload)
            metrics["identity_score"] = round(identity_score, 4) if identity_score is not None else None
            metrics["selection_score"] = round(
                self._selection_score(
                    identity_score=identity_score,
                    accessory_score=accessory_similarity,
                    changed_ratio=float(metrics["changed_ratio"]),
                ),
                6,
            )
            mode_reports[mode] = metrics

        selected_mode = (
            self._select_visual_identity_mode(mode_reports)
            if visual_identity_mode
            else max(
                mode_reports,
                key=lambda mode: self._selection_sort_key(mode, mode_reports[mode]),
            )
        )
        selected_metrics = mode_reports[selected_mode]

        original_image_url = candidate.image_url
        candidate.image_url = mode_payloads[selected_mode]
        candidate.metadata["provider_image_url"] = original_image_url
        candidate.metadata["postprocessed"] = selected_mode != "none"
        candidate.metadata["postprocess"] = {
            "status": "selected",
            "candidate_input_ref": str(candidate_ref),
            "resized_to_source": resized_to_source,
            "selection_basis": selection_basis,
            "preserve_accessory_scope": "visual_localized" if visual_identity_mode else "full_accessory",
            "selected_postprocess_mode": selected_mode,
            "raw_candidate_identity_score": mode_reports["none"].get("identity_score"),
            "selected_identity_score": selected_metrics.get("identity_score"),
            "raw_candidate_accessory_score": mode_reports["none"].get("accessory_similarity_after"),
            "selected_accessory_score": selected_metrics.get("accessory_similarity_after"),
            "postprocess_mode_scores": mode_reports,
            **selected_metrics,
        }
        if "accessory_similarity_after" in selected_metrics:
            candidate.metadata["accessory_metric"] = {
                "metric": "masked_source_similarity",
                "score": selected_metrics["accessory_similarity_after"],
                "mask": preprocess.accessory_mask.uri,
            }
        return candidate

    def _apply_mode(
        self,
        mode: str,
        source_image: Image.Image,
        generated_image: Image.Image,
        shared_context: dict[str, Image.Image],
    ) -> tuple[Image.Image, dict[str, object]]:
        blended = generated_image.copy()
        metrics: dict[str, object] = {
            "core_face_mask_ratio": round(self._mask_fill_ratio(shared_context["core_face_mask"]), 4),
            "identity_hard_mask_ratio": round(self._mask_fill_ratio(shared_context["identity_hard_mask"]), 4),
            "feature_lock_mask_ratio": round(self._mask_fill_ratio(shared_context["feature_lock_mask"]), 4),
            "contour_lock_mask_ratio": round(self._mask_fill_ratio(shared_context["contour_lock_mask"]), 4),
            "accessory_mask_ratio": round(self._mask_fill_ratio(shared_context["accessory_mask"]), 4),
        }
        region_policy = shared_context.get("region_policy")

        if mode == "full_identity":
            blended, similarity = self._apply_protective_blend(
                source_image,
                blended,
                shared_context["core_face_mask"],
                source_weight=0.94,
                radius=self._scaled_radius(source_image.size, minimum=4, maximum=18, fraction=0.008),
            )
            if similarity is not None:
                metrics["core_face_similarity_after"] = round(similarity, 4)
            blended, similarity = self._apply_protective_blend(
                source_image,
                blended,
                shared_context["identity_hard_mask"],
                source_weight=0.9,
                radius=self._scaled_radius(source_image.size, minimum=2, maximum=10, fraction=0.004),
            )
            if similarity is not None:
                metrics["identity_hard_similarity_after"] = round(similarity, 4)

        if mode in {"light_identity", "full_identity"}:
            blended, similarity = self._apply_protective_blend(
                source_image,
                blended,
                shared_context["feature_lock_mask"],
                source_weight=0.86 if mode == "light_identity" else 0.9,
                radius=self._scaled_radius(source_image.size, minimum=1, maximum=6, fraction=0.0025),
            )
            if similarity is not None:
                metrics["feature_lock_similarity_after"] = round(similarity, 4)
            blended, similarity = self._apply_protective_blend(
                source_image,
                blended,
                shared_context["contour_lock_mask"],
                source_weight=0.8 if mode == "light_identity" else 0.86,
                radius=self._scaled_radius(source_image.size, minimum=2, maximum=8, fraction=0.003),
            )
            if similarity is not None:
                metrics["contour_lock_similarity_after"] = round(similarity, 4)

        if mode == "region_adaptive" and region_policy is not None:
            blended, similarity = self._apply_protective_blend(
                source_image,
                blended,
                shared_context["core_face_mask"],
                source_weight=region_policy.face_core.source_weight,
                use_luminance_alignment=False,
                radius=self._scaled_radius(source_image.size, minimum=4, maximum=18, fraction=0.008),
            )
            if similarity is not None:
                metrics["core_face_similarity_after"] = round(similarity, 4)
            blended, similarity = self._apply_protective_blend(
                source_image,
                blended,
                shared_context["identity_hard_mask"],
                source_weight=min(1.0, region_policy.face_core.source_weight + 0.02),
                use_luminance_alignment=False,
                radius=self._scaled_radius(source_image.size, minimum=2, maximum=10, fraction=0.004),
            )
            if similarity is not None:
                metrics["identity_hard_similarity_after"] = round(similarity, 4)
            blended, similarity = self._apply_protective_blend(
                source_image,
                blended,
                shared_context["feature_lock_mask"],
                source_weight=region_policy.feature_lock.source_weight,
                use_luminance_alignment=False,
                radius=self._scaled_radius(source_image.size, minimum=1, maximum=6, fraction=0.0025),
            )
            if similarity is not None:
                metrics["feature_lock_similarity_after"] = round(similarity, 4)
            blended, similarity = self._apply_protective_blend(
                source_image,
                blended,
                shared_context["contour_lock_mask"],
                source_weight=region_policy.contour.source_weight,
                use_luminance_alignment=False,
                radius=self._scaled_radius(source_image.size, minimum=2, maximum=8, fraction=0.003),
            )
            if similarity is not None:
                metrics["contour_lock_similarity_after"] = round(similarity, 4)
            metrics["region_policy"] = region_policy.model_dump(mode="json")

        if mode in {"accessory_only", "light_identity", "full_identity"}:
            if shared_context.get("visual_identity_mode"):
                blended, similarity = self._apply_protective_blend(
                    source_image,
                    blended,
                    shared_context["preserve_accessory_mask"],
                    source_weight=1.0,
                    use_luminance_alignment=False,
                    radius=self._scaled_radius(source_image.size, minimum=1, maximum=4, fraction=0.002),
                )
                if similarity is not None:
                    metrics["accessory_similarity_after"] = round(similarity, 4)
            else:
                blended = Image.composite(source_image, blended, shared_context["preserve_accessory_mask"])
        if mode == "region_adaptive" and region_policy is not None:
            if region_policy.accessory.source_weight >= 0.99:
                blended = Image.composite(source_image, blended, shared_context["preserve_accessory_mask"])
            else:
                blended, similarity = self._apply_protective_blend(
                    source_image,
                    blended,
                    shared_context["preserve_accessory_mask"],
                    source_weight=region_policy.accessory.source_weight,
                    radius=self._scaled_radius(source_image.size, minimum=1, maximum=5, fraction=0.0025),
                )
                if similarity is not None:
                    metrics["accessory_similarity_after"] = round(similarity, 4)

        return blended, metrics

    def _apply_protective_blend(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        mask: Image.Image,
        *,
        source_weight: float,
        use_luminance_alignment: bool = True,
        radius: int,
    ) -> tuple[Image.Image, float | None]:
        mask_ratio = self._mask_fill_ratio(mask)
        if mask_ratio <= 0.001:
            return target_image, None
        source_aligned = (
            self._align_masked_luminance(source_image, target_image, mask)
            if use_luminance_alignment
            else source_image
        )
        source_weight = max(0.0, min(1.0, float(source_weight)))
        if source_weight <= 0.0:
            return target_image, self._masked_similarity(source_image, target_image, mask)
        protected_region = self._mix_images(
            base_image=target_image,
            overlay_image=source_aligned,
            overlay_weight=source_weight,
        )
        feather_mask = self._feather_mask(mask, radius=radius)
        blended = Image.composite(protected_region, target_image, feather_mask)
        similarity = self._masked_similarity(source_image, blended, mask)
        return blended, similarity

    def _can_score_identity(self, preprocess: PreprocessResult) -> bool:
        return bool(
            preprocess.id_embedding.dimension > 0
            and preprocess.id_embedding.vector
            and preprocess.id_embedding.provider != "pseudo_preview"
            and face_identity_service.available
        )

    def _compute_identity_score(
        self,
        preprocess: PreprocessResult,
        candidate_image_ref: str,
    ) -> float | None:
        if not self._can_score_identity(preprocess):
            return None
        return face_identity_service.compare_embedding_to_image(
            preprocess.id_embedding,
            candidate_image_ref,
        )

    def _selection_score(
        self,
        *,
        identity_score: float | None,
        accessory_score: float | None,
        changed_ratio: float,
    ) -> float:
        if identity_score is not None:
            return (
                identity_score
                + 0.01 * max(0.0, accessory_score or 0.0)
                - 0.001 * max(0.0, changed_ratio)
            )
        return max(0.0, accessory_score or 0.0) - 0.05 * max(0.0, changed_ratio)

    def _selection_sort_key(
        self,
        mode: str,
        report: dict[str, object],
    ) -> tuple[float, float, float, float, float]:
        selection_score = float(report.get("selection_score") or 0.0)
        identity_score = report.get("identity_score")
        accessory_score = float(report.get("accessory_similarity_after") or 0.0)
        changed_ratio = float(report.get("changed_ratio") or 0.0)
        identity_key = float(identity_score) if isinstance(identity_score, (int, float)) else -1.0
        return (
            selection_score,
            identity_key,
            accessory_score,
            -changed_ratio,
            -float(self._MODE_PRIORITY.get(mode, 999)),
        )

    def _select_visual_identity_mode(self, mode_reports: dict[str, dict[str, object]]) -> str:
        none_report = mode_reports.get("none", {})
        accessory_report = mode_reports.get("accessory_only", {})
        none_identity = float(none_report.get("identity_score") or -1.0)
        accessory_identity = float(accessory_report.get("identity_score") or -1.0)
        none_accessory = float(none_report.get("accessory_similarity_after") or 0.0)
        accessory_accessory = float(accessory_report.get("accessory_similarity_after") or 0.0)

        if (
            accessory_accessory - none_accessory >= 0.20
            and accessory_identity >= none_identity - 0.012
        ):
            return "accessory_only"

        return max(
            mode_reports,
            key=lambda mode: self._selection_sort_key(mode, mode_reports[mode]),
        )

    def _load_rgb(self, image_ref: str) -> Image.Image:
        return Image.open(io.BytesIO(load_image_bytes(image_ref))).convert("RGB")

    def _load_mask(self, mask_ref: str, size: tuple[int, int]) -> Image.Image:
        mask = Image.open(io.BytesIO(load_image_bytes(mask_ref))).convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        return self._binarize(mask)

    def _build_core_face_protect_mask(
        self,
        face_lock_mask: Image.Image,
        makeup_mask: Image.Image,
        accessory_mask: Image.Image,
    ) -> Image.Image:
        makeup_halo = self._dilate_mask(
            makeup_mask,
            radius=self._scaled_radius(face_lock_mask.size, minimum=4, maximum=14, fraction=0.007),
        )
        accessory_halo = self._dilate_mask(
            accessory_mask,
            radius=self._scaled_radius(face_lock_mask.size, minimum=2, maximum=6, fraction=0.0035),
        )
        core = ImageChops.subtract(face_lock_mask, makeup_halo)
        core = ImageChops.subtract(core, accessory_halo)
        core = self._erode_mask(core, 1)
        return self._binarize(core)

    def _build_identity_hard_protect_mask(
        self,
        face_lock_mask: Image.Image,
        feature_lock_mask: Image.Image,
        contour_lock_mask: Image.Image,
        makeup_mask: Image.Image,
        hair_mask: Image.Image,
        accessory_mask: Image.Image,
    ) -> Image.Image:
        makeup_exclusion = self._dilate_mask(makeup_mask, 2)
        hair_exclusion = self._dilate_mask(hair_mask, 2)
        accessory_exclusion = self._dilate_mask(accessory_mask, 2)
        hard = self._union_masks(
            [
                self._erode_mask(face_lock_mask, 2),
                self._erode_mask(feature_lock_mask, 1),
                self._erode_mask(contour_lock_mask, 1),
            ]
        )
        hard = ImageChops.subtract(hard, makeup_exclusion)
        hard = ImageChops.subtract(hard, hair_exclusion)
        hard = ImageChops.subtract(hard, accessory_exclusion)
        return self._binarize(hard, threshold=48)

    def _build_visual_accessory_protect_mask(
        self,
        accessory_mask: Image.Image,
        face_lock_mask: Image.Image,
        feature_lock_mask: Image.Image,
    ) -> Image.Image:
        feature_halo = self._dilate_mask(
            feature_lock_mask,
            radius=self._scaled_radius(feature_lock_mask.size, minimum=12, maximum=28, fraction=0.018),
        )
        face_halo = self._dilate_mask(
            face_lock_mask,
            radius=self._scaled_radius(face_lock_mask.size, minimum=2, maximum=8, fraction=0.0045),
        )
        localized_accessory = ImageChops.multiply(accessory_mask, feature_halo)
        localized_accessory = ImageChops.multiply(localized_accessory, face_halo)
        bbox = self._mask_bbox(face_lock_mask)
        if bbox is None:
            return self._binarize(localized_accessory, threshold=24)
        x1, y1, x2, y2 = bbox
        face_height = max(1, y2 - y1)
        lower_limit = min(face_lock_mask.size[1], int(round(y1 + face_height * 0.72)))
        localized_np = np.array(localized_accessory, dtype=np.uint8)
        if lower_limit < localized_np.shape[0]:
            localized_np[lower_limit:, :] = 0
        return self._binarize(Image.fromarray(localized_np, mode="L"), threshold=24)

    def _union_masks(self, masks: list[Image.Image]) -> Image.Image:
        result = Image.new("L", masks[0].size, 0)
        for mask in masks:
            result = ImageChops.lighter(result, mask.convert("L"))
        return result

    def _align_masked_luminance(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        source_np = np.array(source_image, dtype=np.float32)
        target_np = np.array(target_image, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.uint8) > 0
        if not mask_np.any():
            return source_image

        source_luma = np.dot(source_np[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        target_luma = np.dot(target_np[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        delta = float(target_luma[mask_np].mean() - source_luma[mask_np].mean())
        adjusted = np.clip(source_np + delta, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(adjusted, mode="RGB")

    def _mix_images(
        self,
        *,
        base_image: Image.Image,
        overlay_image: Image.Image,
        overlay_weight: float,
    ) -> Image.Image:
        overlay_weight = max(0.0, min(1.0, float(overlay_weight)))
        if overlay_weight <= 0.0:
            return base_image
        if overlay_weight >= 1.0:
            return overlay_image
        base_np = np.array(base_image, dtype=np.float32)
        overlay_np = np.array(overlay_image, dtype=np.float32)
        mixed = np.clip(
            base_np * (1.0 - overlay_weight) + overlay_np * overlay_weight,
            0.0,
            255.0,
        ).astype(np.uint8)
        return Image.fromarray(mixed, mode="RGB")

    def _masked_similarity(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        mask: Image.Image,
    ) -> float | None:
        source_np = np.array(source_image, dtype=np.float32)
        target_np = np.array(target_image, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.uint8) > 0
        if not mask_np.any():
            return None
        diff = np.abs(source_np - target_np).mean(axis=2)
        score = 1.0 - float(diff[mask_np].mean() / 255.0)
        return max(0.0, min(1.0, score))

    def _changed_ratio(self, before: Image.Image, after: Image.Image, threshold: int = 8) -> float:
        before_np = np.array(before, dtype=np.int16)
        after_np = np.array(after, dtype=np.int16)
        changed = np.abs(before_np - after_np).max(axis=2) >= threshold
        return float(changed.mean()) if changed.size else 0.0

    def _mask_fill_ratio(self, mask: Image.Image) -> float:
        mask_np = np.array(mask, dtype=np.uint8)
        return float((mask_np > 0).mean()) if mask_np.size else 0.0

    def _mask_bbox(self, mask: Image.Image) -> tuple[int, int, int, int] | None:
        mask_np = np.array(mask, dtype=np.uint8) > 0
        if not mask_np.any():
            return None
        ys, xs = np.where(mask_np)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _scaled_radius(
        self,
        size: tuple[int, int],
        *,
        minimum: int,
        maximum: int,
        fraction: float,
    ) -> int:
        radius = int(round(max(size) * fraction))
        return max(minimum, min(maximum, radius))

    def _feather_mask(self, mask: Image.Image, radius: int) -> Image.Image:
        if radius <= 0:
            return self._binarize(mask)
        return mask.convert("L").filter(ImageFilter.GaussianBlur(radius=radius))

    def _dilate_mask(self, mask: Image.Image, radius: int) -> Image.Image:
        if radius <= 0:
            return self._binarize(mask)
        return mask.convert("L").filter(ImageFilter.MaxFilter(radius * 2 + 1))

    def _erode_mask(self, mask: Image.Image, radius: int) -> Image.Image:
        if radius <= 0:
            return self._binarize(mask)
        return mask.convert("L").filter(ImageFilter.MinFilter(radius * 2 + 1))

    def _binarize(self, mask: Image.Image, threshold: int = 24) -> Image.Image:
        return mask.convert("L").point(lambda value: 255 if value >= threshold else 0)

    def _to_data_url(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{payload}"


postprocess_service = PostprocessService()
