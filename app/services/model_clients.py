import json
import time
from typing import Any
from urllib import request

from volcengine.visual.VisualService import VisualService

from app.config import settings
from app.models.job import JobRecord
from app.models.pipeline import CandidateResult, PreprocessResult, ReferenceParseResult
from app.utils.images import is_http_url, normalize_image_pair_to_base64


class GenericHTTPImageClient:
    def build_payload(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        candidate_index: int,
    ) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "candidate_index": candidate_index,
            "model": settings.local_inpaint.model_name,
            "mode": job.mode,
            "source_image": job.source_image,
            "reference_image": job.reference_image,
            "preserve_accessories": job.preserve_accessories,
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
    ) -> list[CandidateResult]:
        endpoint = settings.local_inpaint.endpoint
        headers = {"Content-Type": "application/json"}
        if settings.local_inpaint.api_key:
            headers[settings.local_inpaint.auth_header] = settings.local_inpaint.api_key

        results: list[CandidateResult] = []
        for idx in range(job.candidate_count):
            payload = self.build_payload(job, preprocess, reference, idx)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            with request.urlopen(req, timeout=settings.local_inpaint.timeout_seconds) as resp:
                response_text = resp.read().decode("utf-8")
            data = json.loads(response_text)
            results.append(self._parse_response(job.job_id, idx, data))
        return results

    def _parse_response(
        self,
        job_id: str,
        candidate_index: int,
        data: dict[str, Any],
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
        return CandidateResult(
            candidate_id=f"{job_id}_local_{candidate_index}",
            pipeline_type="local_inpaint",
            image_url=image_url,
            metadata=metadata,
        )


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
    ) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        for idx in range(job.candidate_count):
            submit_form = self._build_submit_form(job, preprocess, reference, idx)
            submit_resp = self._service.cv_json_api(settings.ark_http.inpaint_action, submit_form)
            task_id = self._extract_task_id(submit_resp)
            if not task_id:
                raise ValueError("Ark submit response missing task_id")
            result_resp = self._poll_result(job, task_id, idx)
            results.append(self._parse_result(job.job_id, idx, task_id, submit_form, result_resp))
        return results

    def _build_submit_form(
        self,
        job: JobRecord,
        preprocess: PreprocessResult,
        reference: ReferenceParseResult,
        candidate_index: int,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(reference, job)
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

        image_urls: list[str] = []
        binary_data_base64: list[str] = []
        both_remote = is_http_url(job.source_image) and is_http_url(job.reference_image)
        if both_remote:
            image_urls = [job.source_image, job.reference_image]
        else:
            source_b64, reference_b64 = normalize_image_pair_to_base64(
                job.source_image,
                job.reference_image,
                size=(1024, 1024),
            )
            binary_data_base64 = [source_b64, reference_b64]

        if image_urls:
            form["image_urls"] = image_urls
        if binary_data_base64:
            form["binary_data_base64"] = binary_data_base64

        req_json = {
            "job_id": job.job_id,
            "candidate_index": candidate_index,
            "mode": job.mode,
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
            },
            "reference_features": reference.model_dump(mode="json"),
        }
        form["req_json"] = json.dumps(req_json, ensure_ascii=False)
        return form

    def _build_prompt(self, reference: ReferenceParseResult, job: JobRecord) -> str:
        hair = reference.hair_features
        bangs = reference.bangs
        makeup = reference.makeup_features
        texture = reference.texture_features
        parts = [
            "保留原图人物身份、脸型和五官结构",
            "仅迁移参考图的发型、刘海、底妆、腮红、眼妆、修容高光、眉妆、睫毛和唇妆",
            "保留原图眼镜、发箍和非编辑配饰",
            "发型: {style}, {updo}, {length}, {parting}, {texture}, hair color {color}".format(
                style=hair.style,
                updo=hair.updo_type,
                length=hair.length,
                parting=hair.parting,
                texture=hair.texture,
                color=hair.color.label,
            ),
            "刘海: {bangs}".format(bangs=("none" if not bangs.exists else bangs.type)),
            "底妆: finish {finish}, intensity {intensity:.2f}, brightness shift {shift:.2f}, glow {glow:.2f}".format(
                finish=makeup.base_makeup.finish,
                intensity=makeup.base_makeup.intensity,
                shift=makeup.base_makeup.brightness_shift,
                glow=makeup.base_makeup.glow,
            ),
            "腮红: color {color}, intensity {intensity:.2f}".format(
                color=makeup.blush.color,
                intensity=makeup.blush.intensity,
            ),
            "修容高光: contour {contour}, highlight {highlight}".format(
                contour=makeup.contour.color,
                highlight=makeup.highlight.color,
            ),
            "眼妆: eyeshadow {eyeshadow}, eyeliner {eyeliner}, lashes {lashes:.2f}, aegyo sal {aegyo:.2f}".format(
                eyeshadow=makeup.eyeshadow.main_color,
                eyeliner=makeup.eyeliner.style,
                lashes=makeup.eyelashes.intensity,
                aegyo=makeup.aegyo_sal.intensity,
            ),
            "眉毛: shape {shape}, color {color}, intensity {intensity:.2f}".format(
                shape=makeup.eyebrow.shape,
                color=makeup.eyebrow.color,
                intensity=makeup.eyebrow.intensity,
            ),
            "唇妆: color {lips}, gloss {gloss:.2f}, saturation {sat:.2f}".format(
                lips=makeup.lips.color,
                gloss=makeup.lips.gloss,
                sat=makeup.lips.saturation,
            ),
            "整体照片质感: {style}, {vibe}".format(
                style=texture.photo_style,
                vibe=texture.overall_vibe,
            ),
        ]
        if reference.normalized_prompt_tokens:
            parts.append("关键特征: " + ", ".join(reference.normalized_prompt_tokens))
        parts.append("妆容迁移强度 {:.2f}".format(job.makeup_strength))
        parts.append("发型迁移强度 {:.2f}".format(job.hairstyle_strength))
        parts.append("身份锁定强度 {:.2f}".format(job.identity_lock_strength))
        return "；".join(parts)

    def _poll_result(self, job: JobRecord, task_id: str, candidate_index: int) -> dict[str, Any]:
        req_key = settings.ark_http.inpaint_model or settings.local_inpaint.model_name
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
            last_resp = self._service.cv_json_api(settings.ark_http.inpaint_get_action, form)
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
            candidate_id=f"{job_id}_local_{candidate_index}",
            pipeline_type="local_inpaint",
            image_url=image_url,
            metadata={
                "provider_mode": "ark_http",
                "provider_task_id": task_id,
                "provider_submit_req_key": submit_form.get("req_key"),
                "raw_response": response,
                "provider_prompt": submit_form.get("prompt"),
                "provider_request_features": json.loads(submit_form.get("req_json", "{}")),
            },
        )
