import base64
import io
import json
import time
from typing import Any
from urllib import request

from PIL import Image
from volcengine.visual.VisualService import VisualService

from app.config import settings
from app.models.job import JobRecord
from app.models.pipeline import (
    CandidateResult,
    GenerationControlBundle,
    GenerationStrengthControls,
    ReferenceParseResult,
)
from app.utils.images import is_http_url, load_image_bytes, normalize_image_pair_to_base64


StageContext = dict[str, Any]


class GenericHTTPImageClient:
    def build_payload(
        self,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        candidate_index: int,
        stage_context: StageContext | None = None,
    ) -> dict[str, Any]:
        stage_context = stage_context or {}
        source_image = stage_context.get("source_image", control_bundle.source_image)
        edit_target = stage_context.get("edit_target", job.mode)
        active_controls = self._resolve_controls(control_bundle, stage_context)
        active_edit_mask = (
            control_bundle.editable_hair_mask
            if edit_target == "hair_only"
            else control_bundle.editable_makeup_mask
        )
        return {
            "job_id": job.job_id,
            "candidate_index": candidate_index,
            "model": settings.local_inpaint.model_name,
            "mode": job.mode,
            "source_image": source_image,
            "reference_image": control_bundle.reference_image,
            "preserve_accessories": active_controls.preserve_accessories,
            "pipeline_variant": control_bundle.pipeline_variant,
            "delivery_mode": control_bundle.delivery_mode,
            "stage": self._build_stage_payload(job, stage_context),
            "controls": active_controls.model_dump(mode="json"),
            "masks": {
                "id_mask": control_bundle.id_mask.model_dump(mode="json"),
                "style_mask": control_bundle.style_mask.model_dump(mode="json"),
                "accessory_mask": control_bundle.accessory_mask.model_dump(mode="json"),
                "editable_hair_mask": control_bundle.editable_hair_mask.model_dump(mode="json"),
                "editable_makeup_mask": control_bundle.editable_makeup_mask.model_dump(mode="json"),
                "face_lock_mask": control_bundle.face_lock_mask.model_dump(mode="json"),
                "feature_lock_mask": control_bundle.feature_lock_mask.model_dump(mode="json"),
                "contour_lock_mask": control_bundle.contour_lock_mask.model_dump(mode="json"),
                "active_edit_mask": active_edit_mask.model_dump(mode="json"),
            },
            "source_structure": {
                "face_bbox": control_bundle.face_bbox.model_dump(mode="json"),
                "pose": control_bundle.pose.model_dump(mode="json"),
                "landmarks_106": [point.model_dump(mode="json") for point in control_bundle.landmarks_106],
                "face_mesh": control_bundle.face_mesh.model_dump(mode="json"),
            },
            "quality_gate": control_bundle.quality_gate.model_dump(mode="json"),
            "identity_embedding": control_bundle.identity_embedding.model_dump(mode="json"),
            "capability_profile": (
                control_bundle.capability_profile.model_dump(mode="json")
                if control_bundle.capability_profile
                else None
            ),
            "region_gating_policy": (
                control_bundle.region_gating_policy.model_dump(mode="json")
                if control_bundle.region_gating_policy
                else None
            ),
            "reference_features": reference.model_dump(mode="json"),
            "reference_region_assets": reference.region_assets.model_dump(mode="json"),
        }

    def generate_candidates(
        self,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        stage_context: StageContext | None = None,
        pipeline_type: str = "local_inpaint",
    ) -> list[CandidateResult]:
        endpoint = settings.local_inpaint.endpoint
        headers = {"Content-Type": "application/json"}
        if settings.local_inpaint.api_key:
            headers[settings.local_inpaint.auth_header] = settings.local_inpaint.api_key

        results: list[CandidateResult] = []
        for idx in range(job.candidate_count):
            payload = self.build_payload(
                job,
                control_bundle,
                reference,
                idx,
                stage_context=stage_context,
            )
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            with request.urlopen(req, timeout=settings.local_inpaint.timeout_seconds) as resp:
                response_text = resp.read().decode("utf-8")
            data = json.loads(response_text)
            results.append(
                self._parse_response(
                    job.job_id,
                    idx,
                    data,
                    stage_context=stage_context,
                    pipeline_type=pipeline_type,
                )
            )
        return results

    def _parse_response(
        self,
        job_id: str,
        candidate_index: int,
        data: dict[str, Any],
        stage_context: StageContext | None = None,
        pipeline_type: str = "local_inpaint",
    ) -> CandidateResult:
        image_url = None
        metadata: dict[str, Any] = {"provider_mode": "generic_http"}

        if isinstance(data.get("images"), list) and data["images"]:
            image = data["images"][0]
            image_url = image.get("url") or image.get("image_url")
            metadata.update(image.get("metadata", {}))
        elif data.get("result_image"):
            image_url = data.get("result_image")
        elif data.get("image_url"):
            image_url = data.get("image_url")

        if not image_url:
            raise ValueError("Image provider response missing result image url")

        metadata["raw_response"] = data
        metadata["stage_context"] = stage_context or {}
        return CandidateResult(
            candidate_id=f"{job_id}_local_{candidate_index}",
            pipeline_type=pipeline_type,
            image_url=image_url,
            metadata=metadata,
        )

    def _build_stage_payload(
        self,
        job: JobRecord,
        stage_context: StageContext,
    ) -> dict[str, Any]:
        return {
            "name": stage_context.get("stage_name", job.mode),
            "edit_target": stage_context.get("edit_target", job.mode),
            "face_lock": True,
            "stage_index": stage_context.get("stage_index", 0),
            "parent_candidate_id": stage_context.get("parent_candidate_id"),
            "source_stage_candidate_id": stage_context.get("source_stage_candidate_id"),
            "control_overrides": stage_context.get("control_overrides"),
            "region_gating_policy": stage_context.get("region_gating_policy"),
        }


class ArkVisualClient:
    def __init__(self) -> None:
        self._service = VisualService()
        self._service.set_ak(settings.ark_http.access_key)
        self._service.set_sk(settings.ark_http.secret_key)
        self._service.set_connection_timeout(int(settings.ark_http.timeout_seconds))
        self._service.set_socket_timeout(int(settings.ark_http.timeout_seconds))

    def generate_candidates(
        self,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        stage_context: StageContext | None = None,
        pipeline_type: str = "local_inpaint",
    ) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        for idx in range(job.candidate_count):
            submit_form = self._build_submit_form(
                job,
                control_bundle,
                reference,
                idx,
                stage_context=stage_context,
            )
            submit_resp = self._service.cv_json_api(settings.ark_http.inpaint_action, submit_form)
            task_id = self._extract_task_id(submit_resp)
            if not task_id:
                raise ValueError("Ark submit response missing task_id")
            result_resp = self._poll_result(job, task_id, idx)
            results.append(
                self._parse_result(
                    job.job_id,
                    idx,
                    task_id,
                    submit_form,
                    result_resp,
                    stage_context=stage_context,
                    pipeline_type=pipeline_type,
                )
            )
        return results

    def generate_mainline_candidates(
        self,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        pipeline_type: str = "ark_native_control_mainline",
    ) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        for idx in range(job.candidate_count):
            submit_form = self._build_mainline_submit_form(job, control_bundle, reference, idx)
            submit_resp = self._service.cv_json_api(self._resolve_mainline_submit_action(), submit_form)
            task_id = self._extract_task_id(submit_resp)
            if not task_id:
                raise ValueError("Ark mainline submit response missing task_id")
            result_resp = self._poll_result(
                job,
                task_id,
                idx,
                req_key=settings.ark_http.model,
                get_action=self._resolve_mainline_get_action(),
            )
            results.append(
                self._parse_result(
                    job.job_id,
                    idx,
                    task_id,
                    submit_form,
                    result_resp,
                    pipeline_type=pipeline_type,
                    provider_mode=(
                        "ark_http_mainline_native"
                        if pipeline_type == "ark_native_control_mainline"
                        else "ark_http_mainline_hybrid_base"
                    ),
                )
            )
        return results

    def _build_submit_form(
        self,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        candidate_index: int,
        stage_context: StageContext | None = None,
    ) -> dict[str, Any]:
        stage_context = stage_context or {}
        active_controls = self._resolve_controls(control_bundle, stage_context)
        prompt = self._build_prompt(
            reference,
            job,
            control_bundle,
            active_controls,
            stage_context=stage_context,
        )
        req_key = settings.ark_http.inpaint_model or settings.local_inpaint.model_name
        form: dict[str, Any] = {
            "req_key": req_key,
            "prompt": prompt,
            "return_url": True,
            "seed": -1,
            "logo_info": {
                "add_logo": not settings.ark_http.disable_logo,
                "position": 0,
            },
        }

        source_image = stage_context.get("source_image", control_bundle.source_image)
        edit_target = stage_context.get("edit_target", job.mode)
        active_edit_mask = (
            control_bundle.editable_hair_mask.uri
            if edit_target == "hair_only"
            else control_bundle.editable_makeup_mask.uri
        )

        image_urls: list[str] = []
        binary_data_base64: list[str] = []
        source_remote = is_http_url(source_image)
        mask_remote = is_http_url(active_edit_mask)
        if source_remote and mask_remote:
            image_urls = [source_image, active_edit_mask]
        else:
            source_b64, mask_b64 = self._normalize_source_and_mask_to_base64(
                source_image,
                active_edit_mask,
            )
            binary_data_base64 = [source_b64, mask_b64]

        if image_urls:
            form["image_urls"] = image_urls
        if binary_data_base64:
            form["binary_data_base64"] = binary_data_base64

        req_json = {
            "job_id": job.job_id,
            "candidate_index": candidate_index,
            "mode": job.mode,
            "source_image": source_image,
            "reference_image": control_bundle.reference_image,
            "pipeline_variant": control_bundle.pipeline_variant,
            "delivery_mode": control_bundle.delivery_mode,
            "stage": {
                "name": stage_context.get("stage_name", job.mode),
                "edit_target": edit_target,
                "face_lock": True,
                "stage_index": stage_context.get("stage_index", 0),
                "parent_candidate_id": stage_context.get("parent_candidate_id"),
                "source_stage_candidate_id": stage_context.get("source_stage_candidate_id"),
                "control_overrides": stage_context.get("control_overrides"),
            },
            "controls": active_controls.model_dump(mode="json"),
            "masks": {
                "id_mask": control_bundle.id_mask.uri,
                "style_mask": control_bundle.style_mask.uri,
                "accessory_mask": control_bundle.accessory_mask.uri,
                "editable_hair_mask": control_bundle.editable_hair_mask.uri,
                "editable_makeup_mask": control_bundle.editable_makeup_mask.uri,
                "face_lock_mask": control_bundle.face_lock_mask.uri,
                "feature_lock_mask": control_bundle.feature_lock_mask.uri,
                "contour_lock_mask": control_bundle.contour_lock_mask.uri,
                "active_edit_mask": active_edit_mask,
            },
            "source_structure": {
                "face_bbox": control_bundle.face_bbox.model_dump(mode="json"),
                "pose": control_bundle.pose.model_dump(mode="json"),
                "landmarks_106": [point.model_dump(mode="json") for point in control_bundle.landmarks_106],
                "face_mesh": control_bundle.face_mesh.model_dump(mode="json"),
            },
            "identity_embedding": control_bundle.identity_embedding.model_dump(mode="json"),
            "quality_gate": control_bundle.quality_gate.model_dump(mode="json"),
            "region_gating_policy": (
                control_bundle.region_gating_policy.model_dump(mode="json")
                if control_bundle.region_gating_policy
                else None
            ),
            "reference_features": reference.model_dump(mode="json"),
            "reference_region_assets": reference.region_assets.model_dump(mode="json"),
            "inpaint_transport": {
                "image_slot_1": "source_image",
                "image_slot_2": "active_edit_mask",
            },
        }
        form["req_json"] = json.dumps(req_json, ensure_ascii=False)
        return form

    def _build_mainline_submit_form(
        self,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
        reference: ReferenceParseResult,
        candidate_index: int,
    ) -> dict[str, Any]:
        prompt = self._build_mainline_prompt(job, reference, control_bundle)
        form: dict[str, Any] = {
            "req_key": settings.ark_http.model,
            "prompt": prompt,
            "return_url": True,
            "seed": -1,
            "logo_info": {
                "add_logo": not settings.ark_http.disable_logo,
                "position": 0,
            },
        }

        source_image = control_bundle.source_image
        reference_image = control_bundle.reference_image
        source_remote = is_http_url(source_image)
        reference_remote = is_http_url(reference_image)
        if source_remote and reference_remote:
            form["image_urls"] = [source_image, reference_image]
        else:
            source_b64, reference_b64 = normalize_image_pair_to_base64(source_image, reference_image)
            form["binary_data_base64"] = [source_b64, reference_b64]

        req_json = {
            "job_id": job.job_id,
            "candidate_index": candidate_index,
            "mode": job.mode,
            "source_image": source_image,
            "reference_image": reference_image,
            "pipeline_variant": control_bundle.pipeline_variant,
            "delivery_mode": control_bundle.delivery_mode,
            "controls": control_bundle.controls.model_dump(mode="json"),
            "masks": {
                "id_mask": control_bundle.id_mask.uri,
                "style_mask": control_bundle.style_mask.uri,
                "accessory_mask": control_bundle.accessory_mask.uri,
                "editable_hair_mask": control_bundle.editable_hair_mask.uri,
                "editable_makeup_mask": control_bundle.editable_makeup_mask.uri,
                "face_lock_mask": control_bundle.face_lock_mask.uri,
                "feature_lock_mask": control_bundle.feature_lock_mask.uri,
                "contour_lock_mask": control_bundle.contour_lock_mask.uri,
            },
            "source_structure": {
                "face_bbox": control_bundle.face_bbox.model_dump(mode="json"),
                "pose": control_bundle.pose.model_dump(mode="json"),
                "landmarks_106": [point.model_dump(mode="json") for point in control_bundle.landmarks_106],
                "face_mesh": control_bundle.face_mesh.model_dump(mode="json"),
            },
            "identity_embedding": control_bundle.identity_embedding.model_dump(mode="json"),
            "quality_gate": control_bundle.quality_gate.model_dump(mode="json"),
            "region_gating_policy": (
                control_bundle.region_gating_policy.model_dump(mode="json")
                if control_bundle.region_gating_policy
                else None
            ),
            "capability_profile": (
                control_bundle.capability_profile.model_dump(mode="json")
                if control_bundle.capability_profile
                else None
            ),
            "reference_features": reference.model_dump(mode="json"),
            "reference_region_assets": reference.region_assets.model_dump(mode="json"),
            "mainline_transport": {
                "image_slot_1": "source_image",
                "image_slot_2": "reference_image",
                "source_role": "identity_source",
                "reference_role": "makeup_hairstyle_reference",
            },
        }
        form["req_json"] = json.dumps(req_json, ensure_ascii=False)
        return form

    def _build_prompt(
        self,
        reference: ReferenceParseResult,
        job: JobRecord,
        control_bundle: GenerationControlBundle,
        active_controls: GenerationStrengthControls,
        stage_context: StageContext | None = None,
    ) -> str:
        stage_context = stage_context or {}
        edit_target = stage_context.get("edit_target", job.mode)
        stage_name = stage_context.get("stage_name", edit_target)

        hair = reference.hair_features
        bangs = reference.bangs
        makeup = reference.makeup_features
        texture = reference.texture_features
        hair_target = self._join_known_traits(
            [
                ("primary", hair.primary_style),
                ("secondary", hair.secondary_style),
                ("updo", hair.updo_type),
                ("bun silhouette", hair.bun_silhouette),
                ("length", hair.length),
                ("parting", hair.parting),
                ("texture", hair.texture),
                ("surface finish", hair.surface_finish),
                ("color", hair.color.label),
                ("temperature", hair.color_temperature),
                ("depth", hair.color_depth),
            ]
        )
        bangs_target = self._join_known_traits(
            [
                ("type", bangs.type if bangs.exists else "none"),
                ("length", bangs.length),
                ("curve", bangs.curve),
            ]
        )
        brow_target = self._join_known_traits(
            [
                ("shape", makeup.eyebrow.shape),
                ("color", makeup.eyebrow.color),
                ("tone", makeup.eyebrow.tone),
            ]
        )
        eye_target = self._join_known_traits(
            [
                ("upper lid", makeup.eyeshadow.upper_lid_color),
                ("lower lid", makeup.eyeshadow.lower_lid_color),
                ("outer corner", makeup.eyeshadow.outer_corner_color),
                ("main shadow", makeup.eyeshadow.main_color),
                ("secondary shadow", makeup.eyeshadow.secondary_color),
                ("eyeshadow finish", makeup.eyeshadow.finish),
                ("eyeliner", makeup.eyeliner.style),
                ("eyeliner color", makeup.eyeliner.color),
            ]
        )
        lip_target = self._join_known_traits(
            [
                ("color", makeup.lips.color),
                ("finish", makeup.lips.finish),
                ("temperature", makeup.lips.temperature),
                ("lightness", makeup.lips.lightness),
                ("shape", makeup.lips.shape),
            ]
        )
        hair_target = hair_target or "preserve the reference hair structure cues"
        bangs_target = bangs_target or "none"
        brow_target = brow_target or "keep the original brow geometry with only soft color adaptation"
        eye_target = eye_target or "keep eye makeup subtle and source-structure-safe"
        lip_target = lip_target or "keep lip makeup gentle and source-shape-safe"

        parts = [
            f"Stage: {stage_name}",
            "Treat the reference image as a style board for hair and makeup only. Do not copy the reference person's facial anatomy.",
            "Preserve the source person's identity, face shape, eyes, nose, mouth, skin tone base, and age impression.",
            "Lock the face and all accessories. Do not inherit the reference person's facial identity.",
            "If the reference face conflicts with the source face, keep the source face and transfer only hairstyle and makeup attributes.",
            "Preserve glasses, headband, earrings, and non-edit clothing edges.",
            self._build_region_gate_text(control_bundle),
        ]

        if edit_target == "hair_only":
            parts.extend(
                [
                    "Edit only the hair region. Keep facial makeup and face geometry unchanged.",
                    "Do not redraw eyes, nose, lips, brows, eyelids, cheeks, jawline, or skin texture.",
                    "Transfer only hairstyle structure, hairline, bangs, side locks, hair texture, and hair color from the reference.",
                    f"Hair structure target: {hair_target}.",
                    "Hair metrics: crown volume {crown:.2f}, side volume {side:.2f}, hairline exposure {hairline:.2f}, sleekness {sleek:.2f}, gloss {gloss:.2f}".format(
                        crown=hair.volume_crown,
                        side=hair.volume_side,
                        hairline=hair.hairline_exposure,
                        sleek=hair.sleekness,
                        gloss=hair.gloss,
                    ),
                    f"Bangs target: {bangs_target}, density {bangs.density:.2f}, gap ratio {bangs.gap_ratio:.2f}.",
                    "Do not modify lipstick, blush, eyeliner, eyeshadow, contour, highlight, brows, or foundation in this stage.",
                    "Hairstyle transfer strength {:.2f}".format(active_controls.hairstyle_strength),
                ]
            )
            if hair.style == "updo" or "updo" in hair.primary_style:
                parts.append(
                    "Hair shape must remain a lifted bun or updo silhouette on the crown, not loose hair."
                )
            if bangs.exists:
                parts.append(
                    "Keep front bangs covering part of the forehead at eyebrow-to-eye length, with light face-framing side locks when visible in the reference."
                )
        elif edit_target == "makeup_only":
            parts.extend(
                [
                    "Edit only the makeup region. Keep the incoming hairstyle and the face geometry unchanged.",
                    "Do not redraw eye shape, eyebrow shape, nose contour, lip outline geometry, cheeks shape, jawline, or face identity.",
                    "Transfer only makeup semantics from the reference: base, blush, contour, highlight, brows, eyeshadow, eyeliner, lashes, aegyo sal, lips.",
                    "Base makeup: finish {finish}, intensity {intensity:.2f}, coverage {coverage:.2f}, brightness shift {shift:.2f}, glow {glow:.2f}, powderiness {powder:.2f}".format(
                        finish=makeup.base_makeup.finish,
                        intensity=makeup.base_makeup.intensity,
                        coverage=makeup.base_makeup.coverage,
                        shift=makeup.base_makeup.brightness_shift,
                        glow=makeup.base_makeup.glow,
                        powder=makeup.base_makeup.powderiness,
                    ),
                    "Blush / contour / highlight: blush {blush} {blush_intensity:.2f}, contour {contour} {contour_intensity:.2f}, highlight {highlight} {highlight_intensity:.2f}".format(
                        blush=makeup.blush.color,
                        blush_intensity=makeup.blush.intensity,
                        contour=makeup.contour.color,
                        contour_intensity=makeup.contour.intensity,
                        highlight=makeup.highlight.color,
                        highlight_intensity=makeup.highlight.intensity,
                    ),
                    f"Brows target: {brow_target}, thickness {makeup.eyebrow.thickness:.2f}, density {makeup.eyebrow.density:.2f}, hair texture {makeup.eyebrow.hair_texture:.2f}.",
                    f"Eye makeup target: {eye_target}. Keep upper lid, lower lid, and outer-corner color separation instead of collapsing them into one flat shadow wash.",
                    "Eye detail metrics: shimmer {shimmer:.2f}, liner length {liner_len:.2f}, liner thickness {liner_thick:.2f}, lashes intensity {lashes:.2f}, lashes curl {curl:.2f}, aegyo sal {aegyo:.2f}".format(
                        shimmer=makeup.eyeshadow.shimmer,
                        liner_len=makeup.eyeliner.length,
                        liner_thick=makeup.eyeliner.thickness,
                        lashes=makeup.eyelashes.intensity,
                        curl=makeup.eyelashes.curl,
                        aegyo=makeup.aegyo_sal.intensity,
                    ),
                    f"Lip target: {lip_target}, gloss {makeup.lips.gloss:.2f}, saturation {makeup.lips.saturation:.2f}, edge definition {makeup.lips.edge_definition:.2f}, cupid bow {makeup.lips.cupid_bow_definition:.2f}, bite effect {makeup.lips.bite_effect:.2f}.",
                    "Do not change hairstyle structure or hair color in this stage.",
                    "Makeup transfer strength {:.2f}".format(active_controls.makeup_strength),
                ]
            )
        else:
            parts.extend(
                [
                    "Transfer only the reference hairstyle and makeup while preserving identity.",
                    "Do not replace facial identity, eye shape, nose shape, lip geometry, cheek volume, or jawline.",
                    f"Hair target: {hair_target}.",
                    f"Makeup target: brows {brow_target}; eyes {eye_target}; lips {lip_target}.",
                    "Base makeup finish {finish}, overall vibe {vibe}".format(
                        finish=makeup.base_makeup.finish,
                        vibe=texture.overall_vibe,
                    ),
                ]
            )

        parts.append(
            "Overall texture: photo style {style}, vibe {vibe}, caption {caption}".format(
                style=texture.photo_style,
                vibe=texture.overall_vibe,
                caption=reference.style_caption,
            )
        )
        if reference.normalized_prompt_tokens:
            parts.append("Key traits: " + ", ".join(reference.normalized_prompt_tokens))
        if reference.consistency_flags:
            parts.append("Consistency hints: " + ", ".join(reference.consistency_flags))
        parts.append("Identity lock strength {:.2f}".format(active_controls.identity_lock_strength))
        return ". ".join(parts)

    def _build_mainline_prompt(
        self,
        job: JobRecord,
        reference: ReferenceParseResult,
        control_bundle: GenerationControlBundle,
    ) -> str:
        hair = reference.hair_features
        bangs = reference.bangs
        makeup = reference.makeup_features
        texture = reference.texture_features
        hair_target = self._join_known_traits(
            [
                ("primary", hair.primary_style),
                ("secondary", hair.secondary_style),
                ("updo", hair.updo_type),
                ("bun silhouette", hair.bun_silhouette),
                ("parting", hair.parting),
                ("texture", hair.texture),
                ("surface finish", hair.surface_finish),
                ("hair color", hair.color.label),
                ("temperature", hair.color_temperature),
                ("depth", hair.color_depth),
            ]
        )
        eye_target = self._join_known_traits(
            [
                ("upper lid", makeup.eyeshadow.upper_lid_color),
                ("lower lid", makeup.eyeshadow.lower_lid_color),
                ("outer corner", makeup.eyeshadow.outer_corner_color),
                ("eyeshadow finish", makeup.eyeshadow.finish),
                ("eyeliner", makeup.eyeliner.style),
                ("eyeliner color", makeup.eyeliner.color),
            ]
        )
        lip_target = self._join_known_traits(
            [
                ("lip color", makeup.lips.color),
                ("lip finish", makeup.lips.finish),
                ("lip temperature", makeup.lips.temperature),
                ("lip lightness", makeup.lips.lightness),
            ]
        )
        brow_target = self._join_known_traits(
            [
                ("brow shape", makeup.eyebrow.shape),
                ("brow color", makeup.eyebrow.color),
                ("brow tone", makeup.eyebrow.tone),
            ]
        )
        hair_target = hair_target or "preserve the reference hair structure cues"
        eye_target = eye_target or "keep eye makeup subtle and source-structure-safe"
        lip_target = lip_target or "keep lip makeup gentle and source-shape-safe"
        brow_target = brow_target or "keep the original brow geometry with only soft color adaptation"

        parts = [
            "Use the first input image as the source identity image and the second input image as the hairstyle and makeup reference image.",
            "Treat the second input image as a non-identity style board: read hairstyle silhouette, bang layout, brow tone, eye color placement, lip color, and finish, but ignore the reference person's facial anatomy.",
            "Preserve the source person's facial identity, face shape, eyes, nose, mouth, age impression, and core facial structure.",
            "Transfer the reference hairstyle and makeup strongly, while keeping the output as the same person from the source image.",
            "If the reference face conflicts with the source face, keep the source face and transfer only hairstyle and makeup attributes.",
            "Do not redraw the source person's eye shape, eyebrow geometry, nose bridge, lip contour, cheek volume, or jawline.",
            "Preserve glasses, headband, earrings, and source accessories.",
            "Do not inherit the reference person's face, body identity, clothing, or background.",
            self._build_region_gate_text(control_bundle),
            f"Hairstyle target: {hair_target}. Crown volume {hair.volume_crown:.2f}, side volume {hair.volume_side:.2f}, hairline exposure {hair.hairline_exposure:.2f}.",
            "Bangs target: {bangs}, density {density:.2f}, length {length}, gap ratio {gap:.2f}.".format(
                bangs=("none" if not bangs.exists else bangs.type),
                density=bangs.density,
                length=bangs.length,
                gap=bangs.gap_ratio,
            ),
            f"Makeup target: base finish {makeup.base_makeup.finish}, blush {makeup.blush.color}, contour {makeup.contour.color}, highlight {makeup.highlight.color}, {brow_target}, {eye_target}, {lip_target}.",
            "Keep upper lid, lower lid, and outer-corner makeup semantically distinct when the reference indicates different tones or finishes.",
            "Mainline controls: hairstyle strength {:.2f}, makeup strength {:.2f}, identity lock strength {:.2f}.".format(
                job.hairstyle_strength,
                job.makeup_strength,
                job.identity_lock_strength,
            ),
            "Use the provided source face mesh, face bounding box, and landmark guidance as identity-preserving structural anchors when available.",
            "Use the provided reference region assets for hair, bangs, full-eye region, upper lids, lower lids, outer corners, brows, lips, cheeks, and complexion as structured style evidence rather than inheriting the reference identity.",
            "Overall texture: photo style {style}, vibe {vibe}, caption {caption}.".format(
                style=texture.photo_style,
                vibe=texture.overall_vibe,
                caption=reference.style_caption,
            ),
        ]
        if job.identity_mode == "visual_identity":
            parts.extend(
                [
                    "Keep the source person's forehead width, eye spacing, glasses bridge position, nose width, philtrum, mouth width, and chin outline recognizably the same person.",
                    "Allow cosmetic beautification and skin refinement, but keep the source facial geometry visually recognizable at a glance.",
                ]
            )
        if hair.style == "updo" or "updo" in hair.primary_style:
            parts.append(
                "Important hairstyle constraint: keep a lifted bun or updo silhouette on the crown, not loose down hair."
            )
        if bangs.exists:
            parts.append(
                "Important bangs constraint: keep visible front bangs across the forehead around eyebrow-to-eye length, plus thin face-framing side locks when present. Do not expose the whole forehead."
            )
        if reference.normalized_prompt_tokens:
            parts.append("Key traits: " + ", ".join(reference.normalized_prompt_tokens))
        if reference.negative_constraints:
            parts.append("Negative constraints: " + ", ".join(reference.negative_constraints))
        if reference.consistency_flags:
            parts.append("Consistency hints: " + ", ".join(reference.consistency_flags))
        return " ".join(parts)

    def _resolve_controls(
        self,
        control_bundle: GenerationControlBundle,
        stage_context: StageContext | None = None,
    ) -> GenerationStrengthControls:
        stage_context = stage_context or {}
        overrides = stage_context.get("control_overrides")
        if not isinstance(overrides, dict):
            return control_bundle.controls
        return control_bundle.controls.model_copy(update=overrides)

    def _build_region_gate_text(self, control_bundle: GenerationControlBundle) -> str:
        policy = control_bundle.region_gating_policy
        if policy is None:
            return (
                "Regional gate target: keep face and accessories source-dominant, keep hair and makeup "
                "style-dominant, and never let reference identity overwrite source facial structure."
            )
        return (
            "Regional gate target: face core {face_s:.2f}/{face_t:.2f} source-style, "
            "feature lock {feature_s:.2f}/{feature_t:.2f}, contour {contour_s:.2f}/{contour_t:.2f}, "
            "accessory {acc_s:.2f}/{acc_t:.2f}, hair {hair_s:.2f}/{hair_t:.2f}, makeup {makeup_s:.2f}/{makeup_t:.2f}. "
            "Interpret source-dominant regions as identity-preserve zones and style-dominant regions as transfer zones."
        ).format(
            face_s=policy.face_core.source_weight,
            face_t=policy.face_core.style_weight,
            feature_s=policy.feature_lock.source_weight,
            feature_t=policy.feature_lock.style_weight,
            contour_s=policy.contour.source_weight,
            contour_t=policy.contour.style_weight,
            acc_s=policy.accessory.source_weight,
            acc_t=policy.accessory.style_weight,
            hair_s=policy.hair.source_weight,
            hair_t=policy.hair.style_weight,
            makeup_s=policy.makeup.source_weight,
            makeup_t=policy.makeup.style_weight,
        )

    def _is_known_trait(self, value: object) -> bool:
        return value not in {None, "", "unknown", "unclear", "none"}

    def _join_known_traits(self, items: list[tuple[str, object]]) -> str:
        parts = [
            f"{label} {value}"
            for label, value in items
            if self._is_known_trait(value)
        ]
        return ", ".join(parts)

    def _normalize_source_and_mask_to_base64(
        self,
        source_image: str,
        mask_image: str,
        canvas_size: tuple[int, int] = (1024, 1024),
    ) -> tuple[str, str]:
        source = Image.open(io.BytesIO(load_image_bytes(source_image))).convert("RGB")
        mask = Image.open(io.BytesIO(load_image_bytes(mask_image))).convert("L")

        source_fit = self._fit_image(source, canvas_size, mode="RGB", fill=255)
        mask_fit = self._fit_image(mask, canvas_size, mode="L", fill=0)

        source_buffer = io.BytesIO()
        source_fit.save(source_buffer, format="PNG")
        mask_buffer = io.BytesIO()
        mask_fit.save(mask_buffer, format="PNG")
        return (
            base64.b64encode(source_buffer.getvalue()).decode("utf-8"),
            base64.b64encode(mask_buffer.getvalue()).decode("utf-8"),
        )

    def _fit_image(
        self,
        image: Image.Image,
        size: tuple[int, int],
        mode: str,
        fill: int,
    ) -> Image.Image:
        contained = image.copy()
        contained.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new(mode, size, fill)
        x = (size[0] - contained.width) // 2
        y = (size[1] - contained.height) // 2
        canvas.paste(contained, (x, y))
        return canvas

    def _poll_result(
        self,
        job: JobRecord,
        task_id: str,
        candidate_index: int,
        req_key: str | None = None,
        get_action: str | None = None,
    ) -> dict[str, Any]:
        req_key = req_key or settings.ark_http.inpaint_model or settings.local_inpaint.model_name
        get_action = get_action or settings.ark_http.inpaint_get_action
        form = {
            "req_key": req_key,
            "task_id": task_id,
            "req_json": json.dumps(
                {
                    "return_url": True,
                    "job_id": job.job_id,
                    "candidate_index": candidate_index,
                },
                ensure_ascii=False,
            ),
        }
        last_resp: dict[str, Any] | None = None
        for _ in range(settings.ark_http.max_poll_attempts):
            last_resp = self._service.cv_json_api(get_action, form)
            if self._is_task_finished(last_resp):
                return last_resp
            time.sleep(settings.ark_http.poll_interval_seconds)
        raise TimeoutError("Ark task polling exceeded max attempts")

    def _is_task_finished(self, response: dict[str, Any]) -> bool:
        data = response.get("data") or {}
        if data.get("image_urls") or data.get("binary_data_base64"):
            return True
        status = str(data.get("status", "")).lower()
        state = str(data.get("state", "")).lower()
        resp_data = data.get("resp_data")
        if status in {"done", "success", "finished", "succeeded"}:
            return True
        if state in {"done", "success", "finished", "succeeded"}:
            return True
        if isinstance(resp_data, str):
            lowered = resp_data.lower()
            if '"progress": 100' in lowered or '"status":"done"' in lowered:
                return True
        return False

    def _extract_task_id(self, response: dict[str, Any]) -> str | None:
        data = response.get("data") or {}
        return data.get("task_id") or data.get("id")

    def _parse_result(
        self,
        job_id: str,
        candidate_index: int,
        task_id: str,
        submit_form: dict[str, Any],
        response: dict[str, Any],
        stage_context: StageContext | None = None,
        pipeline_type: str = "local_inpaint",
        provider_mode: str = "ark_http",
    ) -> CandidateResult:
        data = response.get("data") or {}
        image_url = None
        if isinstance(data.get("image_urls"), list) and data["image_urls"]:
            image_url = data["image_urls"][0]
        elif data.get("image_url"):
            image_url = data["image_url"]
        elif isinstance(data.get("binary_data_base64"), list) and data["binary_data_base64"]:
            image_url = "data:image/png;base64," + data["binary_data_base64"][0]

        if not image_url:
            raise ValueError("Ark get_result response missing output image")

        candidate_prefix = "global" if pipeline_type.startswith("ark_") or pipeline_type == "global_reference" else "local"

        return CandidateResult(
            candidate_id=f"{job_id}_{candidate_prefix}_{candidate_index}",
            pipeline_type=pipeline_type,
            image_url=image_url,
            metadata={
                "provider_mode": provider_mode,
                "provider_task_id": task_id,
                "provider_submit_req_key": submit_form.get("req_key"),
                "raw_response": response,
                "provider_prompt": submit_form.get("prompt"),
                "provider_request_features": json.loads(submit_form.get("req_json", "{}")),
                "stage_context": stage_context or {},
            },
        )

    def _resolve_mainline_submit_action(self) -> str:
        configured = settings.ark_http.action.strip()
        if configured:
            return configured
        return "CVSync2AsyncSubmitTask"

    def _resolve_mainline_get_action(self) -> str:
        configured = settings.ark_http.get_action.strip()
        if configured:
            return configured
        return "CVSync2AsyncGetResult"
