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
from app.models.pipeline import CandidateResult, PreprocessResult, ReferenceParseResult
from app.utils.images import is_http_url, load_image_bytes, normalize_image_pair_to_base64


StageContext = dict[str, Any]


class GenericHTTPImageClient:
    def build_payload(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        candidate_index: int,
        stage_context: StageContext | None = None,
    ) -> dict[str, Any]:
        stage_context = stage_context or {}
        source_image = stage_context.get("source_image", job.source_image)
        edit_target = stage_context.get("edit_target", job.mode)
        active_edit_mask = (
            preprocess.editable_hair_mask
            if edit_target == "hair_only"
            else preprocess.editable_makeup_mask
        )
        return {
            "job_id": job.job_id,
            "candidate_index": candidate_index,
            "model": settings.local_inpaint.model_name,
            "mode": job.mode,
            "source_image": source_image,
            "reference_image": job.reference_image,
            "preserve_accessories": job.preserve_accessories,
            "stage": self._build_stage_payload(job, stage_context),
            "controls": {
                "makeup_strength": job.makeup_strength,
                "hairstyle_strength": job.hairstyle_strength,
                "identity_lock_strength": job.identity_lock_strength,
            },
            "masks": {
                "id_mask": preprocess.id_mask.model_dump(mode="json"),
                "style_mask": preprocess.style_mask.model_dump(mode="json"),
                "accessory_mask": preprocess.accessory_mask.model_dump(mode="json"),
                "editable_hair_mask": preprocess.editable_hair_mask.model_dump(mode="json"),
                "editable_makeup_mask": preprocess.editable_makeup_mask.model_dump(mode="json"),
                "face_lock_mask": preprocess.face_lock_mask.model_dump(mode="json"),
                "active_edit_mask": active_edit_mask.model_dump(mode="json"),
            },
            "preprocess": {
                "face_bbox": preprocess.face_bbox.model_dump(mode="json"),
                "pose": preprocess.pose.model_dump(mode="json"),
                "accessory_tags": preprocess.accessory_tags,
                "quality_flags": preprocess.quality_flags,
            },
            "reference_features": reference.model_dump(mode="json"),
        }

    def generate_candidates(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        stage_context: StageContext | None = None,
    ) -> list[CandidateResult]:
        endpoint = settings.local_inpaint.endpoint
        headers = {"Content-Type": "application/json"}
        if settings.local_inpaint.api_key:
            headers[settings.local_inpaint.auth_header] = settings.local_inpaint.api_key

        results: list[CandidateResult] = []
        for idx in range(job.candidate_count):
            payload = self.build_payload(
                job,
                preprocess,
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
                )
            )
        return results

    def _parse_response(
        self,
        job_id: str,
        candidate_index: int,
        data: dict[str, Any],
        stage_context: StageContext | None = None,
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
            pipeline_type="local_inpaint",
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
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        stage_context: StageContext | None = None,
    ) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        for idx in range(job.candidate_count):
            submit_form = self._build_submit_form(
                job,
                preprocess,
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
                )
            )
        return results

    def generate_mainline_candidates(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
    ) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        for idx in range(job.candidate_count):
            submit_form = self._build_mainline_submit_form(job, preprocess, reference, idx)
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
                    pipeline_type="ark_complete_mainline",
                    provider_mode="ark_http_mainline",
                )
            )
        return results

    def _build_submit_form(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        candidate_index: int,
        stage_context: StageContext | None = None,
    ) -> dict[str, Any]:
        stage_context = stage_context or {}
        prompt = self._build_prompt(reference, job, stage_context=stage_context)
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

        source_image = stage_context.get("source_image", job.source_image)
        edit_target = stage_context.get("edit_target", job.mode)
        active_edit_mask = (
            preprocess.editable_hair_mask.uri
            if edit_target == "hair_only"
            else preprocess.editable_makeup_mask.uri
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
            "reference_image": job.reference_image,
            "stage": {
                "name": stage_context.get("stage_name", job.mode),
                "edit_target": edit_target,
                "face_lock": True,
                "stage_index": stage_context.get("stage_index", 0),
                "parent_candidate_id": stage_context.get("parent_candidate_id"),
                "source_stage_candidate_id": stage_context.get("source_stage_candidate_id"),
            },
            "controls": {
                "makeup_strength": job.makeup_strength,
                "hairstyle_strength": job.hairstyle_strength,
                "identity_lock_strength": job.identity_lock_strength,
            },
            "masks": {
                "id_mask": preprocess.id_mask.uri,
                "style_mask": preprocess.style_mask.uri,
                "accessory_mask": preprocess.accessory_mask.uri,
                "editable_hair_mask": preprocess.editable_hair_mask.uri,
                "editable_makeup_mask": preprocess.editable_makeup_mask.uri,
                "face_lock_mask": preprocess.face_lock_mask.uri,
                "active_edit_mask": active_edit_mask,
            },
            "reference_features": reference.model_dump(mode="json"),
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
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        candidate_index: int,
    ) -> dict[str, Any]:
        prompt = self._build_mainline_prompt(job, reference)
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

        source_image = job.source_image
        reference_image = job.reference_image
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
            "controls": {
                "makeup_strength": job.makeup_strength,
                "hairstyle_strength": job.hairstyle_strength,
                "identity_lock_strength": job.identity_lock_strength,
                "preserve_accessories": job.preserve_accessories,
            },
            "preprocess": {
                "face_bbox": preprocess.face_bbox.model_dump(mode="json"),
                "pose": preprocess.pose.model_dump(mode="json"),
                "accessory_tags": preprocess.accessory_tags,
                "quality_flags": preprocess.quality_flags,
                "id_embedding_preview": preprocess.id_embedding[:16],
            },
            "masks": {
                "id_mask": preprocess.id_mask.uri,
                "style_mask": preprocess.style_mask.uri,
                "accessory_mask": preprocess.accessory_mask.uri,
                "editable_hair_mask": preprocess.editable_hair_mask.uri,
                "editable_makeup_mask": preprocess.editable_makeup_mask.uri,
                "face_lock_mask": preprocess.face_lock_mask.uri,
            },
            "reference_features": reference.model_dump(mode="json"),
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
        stage_context: StageContext | None = None,
    ) -> str:
        stage_context = stage_context or {}
        edit_target = stage_context.get("edit_target", job.mode)
        stage_name = stage_context.get("stage_name", edit_target)

        hair = reference.hair_features
        bangs = reference.bangs
        makeup = reference.makeup_features
        texture = reference.texture_features

        parts = [
            f"Stage: {stage_name}",
            "Preserve the source person's identity, face shape, eyes, nose, mouth, skin tone base, and age impression.",
            "Lock the face and all accessories. Do not inherit the reference person's facial identity.",
            "Preserve glasses, headband, earrings, and non-edit clothing edges.",
        ]

        if edit_target == "hair_only":
            parts.extend(
                [
                    "Edit only the hair region. Keep facial makeup and face geometry unchanged.",
                    "Transfer only hairstyle structure, hairline, bangs, side locks, hair texture, and hair color from the reference.",
                    "Hair structure: primary {primary}, secondary {secondary}, updo {updo}, length {length}, parting {parting}".format(
                        primary=hair.primary_style,
                        secondary=hair.secondary_style,
                        updo=hair.updo_type,
                        length=hair.length,
                        parting=hair.parting,
                    ),
                    "Hair attributes: texture {texture}, finish {finish}, color {color}, crown volume {crown:.2f}, side volume {side:.2f}, hairline exposure {hairline:.2f}, sleekness {sleek:.2f}, gloss {gloss:.2f}".format(
                        texture=hair.texture,
                        finish=hair.finish,
                        color=hair.color.label,
                        crown=hair.volume_crown,
                        side=hair.volume_side,
                        hairline=hair.hairline_exposure,
                        sleek=hair.sleekness,
                        gloss=hair.gloss,
                    ),
                    "Bangs: {bangs}, density {density:.2f}, length {length}, gap ratio {gap:.2f}".format(
                        bangs=("none" if not bangs.exists else bangs.type),
                        density=bangs.density,
                        length=bangs.length,
                        gap=bangs.gap_ratio,
                    ),
                    "Do not modify lipstick, blush, eyeliner, eyeshadow, contour, highlight, brows, or foundation in this stage.",
                    "Hairstyle transfer strength {:.2f}".format(job.hairstyle_strength),
                ]
            )
        elif edit_target == "makeup_only":
            parts.extend(
                [
                    "Edit only the makeup region. Keep the incoming hairstyle and the face geometry unchanged.",
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
                    "Brows: shape {shape}, color {color}, thickness {thickness:.2f}, hair texture {hair_texture:.2f}".format(
                        shape=makeup.eyebrow.shape,
                        color=makeup.eyebrow.color,
                        thickness=makeup.eyebrow.thickness,
                        hair_texture=makeup.eyebrow.hair_texture,
                    ),
                    "Eye makeup: eyeshadow {main}/{secondary}, eyeliner {eyeliner} {eyeliner_color}, liner length {liner_len:.2f}, liner thickness {liner_thick:.2f}, lashes intensity {lashes:.2f}, lashes curl {curl:.2f}, aegyo sal {aegyo:.2f}".format(
                        main=makeup.eyeshadow.main_color,
                        secondary=makeup.eyeshadow.secondary_color,
                        eyeliner=makeup.eyeliner.style,
                        eyeliner_color=makeup.eyeliner.color,
                        liner_len=makeup.eyeliner.length,
                        liner_thick=makeup.eyeliner.thickness,
                        lashes=makeup.eyelashes.intensity,
                        curl=makeup.eyelashes.curl,
                        aegyo=makeup.aegyo_sal.intensity,
                    ),
                    "Lips: color {lips}, shape {shape}, gloss {gloss:.2f}, saturation {sat:.2f}, edge definition {edge:.2f}, cupid bow {cupid:.2f}, bite effect {bite:.2f}".format(
                        lips=makeup.lips.color,
                        shape=makeup.lips.shape,
                        gloss=makeup.lips.gloss,
                        sat=makeup.lips.saturation,
                        edge=makeup.lips.edge_definition,
                        cupid=makeup.lips.cupid_bow_definition,
                        bite=makeup.lips.bite_effect,
                    ),
                    "Do not change hairstyle structure or hair color in this stage.",
                    "Makeup transfer strength {:.2f}".format(job.makeup_strength),
                ]
            )
        else:
            parts.extend(
                [
                    "Transfer only the reference hairstyle and makeup while preserving identity.",
                    "Hair structure: primary {primary}, secondary {secondary}, updo {updo}, length {length}, parting {parting}".format(
                        primary=hair.primary_style,
                        secondary=hair.secondary_style,
                        updo=hair.updo_type,
                        length=hair.length,
                        parting=hair.parting,
                    ),
                    "Base makeup finish {finish}, lip color {lip}, overall vibe {vibe}".format(
                        finish=makeup.base_makeup.finish,
                        lip=makeup.lips.color,
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
        parts.append("Identity lock strength {:.2f}".format(job.identity_lock_strength))
        return ". ".join(parts)

    def _build_mainline_prompt(
        self,
        job: JobRecord,
        reference: ReferenceParseResult,
    ) -> str:
        hair = reference.hair_features
        bangs = reference.bangs
        makeup = reference.makeup_features
        texture = reference.texture_features

        parts = [
            "Use the first input image as the source identity image and the second input image as the hairstyle and makeup reference image.",
            "Preserve the source person's facial identity, face shape, eyes, nose, mouth, age impression, and core facial structure.",
            "Transfer the reference hairstyle and makeup strongly, while keeping the output as the same person from the source image.",
            "Preserve glasses, headband, earrings, and source accessories.",
            "Do not inherit the reference person's face, body identity, clothing, or background.",
            "Hairstyle target: primary {primary}, secondary {secondary}, updo {updo}, length {length}, parting {parting}, texture {texture}, finish {finish}, color {color}, crown volume {crown:.2f}, side volume {side:.2f}, hairline exposure {hairline:.2f}.".format(
                primary=hair.primary_style,
                secondary=hair.secondary_style,
                updo=hair.updo_type,
                length=hair.length,
                parting=hair.parting,
                texture=hair.texture,
                finish=hair.finish,
                color=hair.color.label,
                crown=hair.volume_crown,
                side=hair.volume_side,
                hairline=hair.hairline_exposure,
            ),
            "Bangs target: {bangs}, density {density:.2f}, length {length}, gap ratio {gap:.2f}.".format(
                bangs=("none" if not bangs.exists else bangs.type),
                density=bangs.density,
                length=bangs.length,
                gap=bangs.gap_ratio,
            ),
            "Makeup target: base finish {finish}, blush {blush}, contour {contour}, highlight {highlight}, brows {browshape}/{browcolor}, eyeshadow {eyemain}/{eyesecondary}, eyeliner {liner}/{linercolor}, lips {lipcolor}.".format(
                finish=makeup.base_makeup.finish,
                blush=makeup.blush.color,
                contour=makeup.contour.color,
                highlight=makeup.highlight.color,
                browshape=makeup.eyebrow.shape,
                browcolor=makeup.eyebrow.color,
                eyemain=makeup.eyeshadow.main_color,
                eyesecondary=makeup.eyeshadow.secondary_color,
                liner=makeup.eyeliner.style,
                linercolor=makeup.eyeliner.color,
                lipcolor=makeup.lips.color,
            ),
            "Mainline controls: hairstyle strength {:.2f}, makeup strength {:.2f}, identity lock strength {:.2f}.".format(
                job.hairstyle_strength,
                job.makeup_strength,
                job.identity_lock_strength,
            ),
            "Overall texture: photo style {style}, vibe {vibe}, caption {caption}.".format(
                style=texture.photo_style,
                vibe=texture.overall_vibe,
                caption=reference.style_caption,
            ),
        ]
        if reference.normalized_prompt_tokens:
            parts.append("Key traits: " + ", ".join(reference.normalized_prompt_tokens))
        if reference.negative_constraints:
            parts.append("Negative constraints: " + ", ".join(reference.negative_constraints))
        if reference.consistency_flags:
            parts.append("Consistency hints: " + ", ".join(reference.consistency_flags))
        return " ".join(parts)

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

        return CandidateResult(
            candidate_id=f"{job_id}_{'global' if pipeline_type == 'ark_complete_mainline' else 'local'}_{candidate_index}",
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
