from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

try:
    import cv2
except Exception:
    cv2 = None

from app.models.pipeline import FaceBBox, LandmarkPoint, MaskAsset, Pose, PreprocessResult
from app.services.reference_parser import reference_parser_service
from app.utils.images import load_image_bytes


class PreprocessService:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self._mask_dir = root / "outputs" / "preprocess_masks"
        self._mask_dir.mkdir(parents=True, exist_ok=True)

    def run(self, source_image: str) -> PreprocessResult:
        image = Image.open(io.BytesIO(load_image_bytes(source_image))).convert("RGB")
        geometry = reference_parser_service._estimate_face_geometry(image)
        detection = reference_parser_service._detect_landmarks(image, geometry)
        raw_masks = reference_parser_service._build_region_masks(image, geometry)

        landmark_masks = None
        if detection is not None:
            landmarks = [
                LandmarkPoint(x=float(point[0]), y=float(point[1]))
                for point in detection.points.tolist()
            ]
            face_bbox = self._face_bbox_from_detection(detection.face_rect, image.size)
            landmark_masks = self._build_landmark_component_masks(image.size, geometry, detection)
        else:
            landmarks = self._fallback_landmarks(geometry)
            face_bbox = self._face_bbox_from_geometry(geometry, image.size)

        face_shape_mask = landmark_masks["face"] if landmark_masks else raw_masks["face"]
        component_masks = landmark_masks or raw_masks
        accessory_mask = self._build_accessory_mask(image, component_masks, geometry)
        hair_mask = self._build_source_hair_mask(
            image,
            geometry,
            raw_masks["hair"],
            face_shape_mask,
            accessory_mask,
            detection,
        )
        makeup_mask = self._build_makeup_mask(component_masks, face_shape_mask, hair_mask, accessory_mask)
        face_lock_mask = self._build_face_lock_mask(face_shape_mask, hair_mask, makeup_mask, accessory_mask)
        style_mask = self._union_masks([hair_mask, makeup_mask])
        id_mask = self._union_masks([face_lock_mask, accessory_mask])

        prefix = self._mask_prefix(source_image)
        hair_path = self._save_mask(hair_mask, f"{prefix}_source_hair_edit_mask.png")
        makeup_path = self._save_mask(makeup_mask, f"{prefix}_source_makeup_edit_mask.png")
        face_lock_path = self._save_mask(face_lock_mask, f"{prefix}_source_face_lock_mask.png")
        accessory_path = self._save_mask(accessory_mask, f"{prefix}_source_accessory_mask.png")
        style_path = self._save_mask(style_mask, f"{prefix}_source_style_mask.png")
        id_path = self._save_mask(id_mask, f"{prefix}_source_id_mask.png")

        quality_flags = ["frontal_face", "strong_identity_lock_recommended", "two_stage_local_edit_ready"]
        if detection is not None:
            quality_flags.append("landmark_driven_masks")
        else:
            quality_flags.append("heuristic_masks")

        accessory_tags = []
        if self._mask_fill_ratio(accessory_mask) > 0.005:
            accessory_tags.append("glasses_or_face_accessory")

        return PreprocessResult(
            face_bbox=face_bbox,
            pose=Pose(yaw=0.0, pitch=0.0, roll=0.0),
            landmarks_106=landmarks,
            id_mask=MaskAsset(kind="id_mask", uri=id_path, width=image.size[0], height=image.size[1]),
            style_mask=MaskAsset(kind="style_mask", uri=style_path, width=image.size[0], height=image.size[1]),
            accessory_mask=MaskAsset(
                kind="accessory_mask",
                uri=accessory_path,
                width=image.size[0],
                height=image.size[1],
            ),
            editable_hair_mask=MaskAsset(
                kind="editable_hair_mask",
                uri=hair_path,
                width=image.size[0],
                height=image.size[1],
            ),
            editable_makeup_mask=MaskAsset(
                kind="editable_makeup_mask",
                uri=makeup_path,
                width=image.size[0],
                height=image.size[1],
            ),
            face_lock_mask=MaskAsset(
                kind="face_lock_mask",
                uri=face_lock_path,
                width=image.size[0],
                height=image.size[1],
            ),
            id_embedding=self._pseudo_identity_embedding(face_lock_mask),
            accessory_tags=accessory_tags,
            quality_flags=quality_flags,
        )

    def _mask_prefix(self, source_image: str) -> str:
        digest = hashlib.md5(source_image.encode("utf-8")).hexdigest()[:12]
        return digest

    def _save_mask(self, mask: Image.Image, filename: str) -> str:
        path = self._mask_dir / filename
        mask.convert("L").save(path)
        return str(path)

    def _face_bbox_from_detection(
        self,
        rect: tuple[int, int, int, int],
        size: tuple[int, int],
    ) -> FaceBBox:
        x, y, w, h = rect
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(size[0], x + w)
        y2 = min(size[1], y + h)
        return FaceBBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _face_bbox_from_geometry(self, geometry, size: tuple[int, int]) -> FaceBBox:
        x1, y1, x2, y2 = reference_parser_service._rect_from_geom(geometry, -0.72, -0.82, 0.72, 1.02)
        return FaceBBox(x1=x1, y1=y1, x2=min(size[0], x2), y2=min(size[1], y2))

    def _fallback_landmarks(self, geometry) -> list[LandmarkPoint]:
        rects = [
            (-0.42, -0.10),
            (0.42, -0.10),
            (0.0, 0.20),
            (-0.26, 0.56),
            (0.26, 0.56),
            (-0.18, -0.34),
            (0.18, -0.34),
            (0.0, -0.62),
        ]
        return [
            LandmarkPoint(
                x=float(geometry.cx + geometry.fw * rx),
                y=float(geometry.cy + geometry.fh * ry),
            )
            for rx, ry in rects
        ]

    def _build_accessory_mask(self, image: Image.Image, masks: dict[str, Image.Image], geometry) -> Image.Image:
        size = image.size
        eye_rect = reference_parser_service._rect_from_geom(geometry, -0.82, -0.10, 0.82, 0.38)
        dark_eye_band = reference_parser_service._create_pixel_mask(
            image,
            eye_rect,
            predicate=lambda rgb, hsv, xy: reference_parser_service._brightness(rgb) < 0.38
            and hsv[1] < 0.30,
        )
        brow_eye_union = self._union_masks(
            [
                masks["eye_band"],
                masks["brow_left"],
                masks["brow_right"],
                masks["liner_left"],
                masks["liner_right"],
            ]
        )
        accessory_seed = ImageChops.multiply(
            dark_eye_band,
            reference_parser_service._dilate_mask(brow_eye_union, 6),
        )
        accessory_seed = reference_parser_service._dilate_mask(accessory_seed, 2)

        top_rect = reference_parser_service._rect_from_geom(geometry, -0.92, -1.18, 0.92, -0.54)
        headband_seed = reference_parser_service._create_pixel_mask(
            image,
            top_rect,
            predicate=lambda rgb, hsv, xy: reference_parser_service._brightness(rgb) < 0.46
            and hsv[1] < 0.28,
        )
        headband_seed = ImageChops.multiply(
            headband_seed,
            reference_parser_service._rect_mask(size, top_rect),
        )
        return self._binarize(
            self._union_masks(
                [
                    accessory_seed,
                    reference_parser_service._erode_mask(headband_seed, 1),
                ]
            )
        )

    def _build_source_hair_mask(
        self,
        image: Image.Image,
        geometry,
        fallback_hair_mask: Image.Image,
        face_mask: Image.Image,
        accessory_mask: Image.Image,
        detection,
    ) -> Image.Image:
        if cv2 is None:
            return self._refine_hair_edit_mask(fallback_hair_mask, face_mask, accessory_mask, geometry)

        max_dim = 512
        scale = min(1.0, max_dim / max(image.size))
        small_size = (
            max(1, int(round(image.size[0] * scale))),
            max(1, int(round(image.size[1] * scale))),
        )
        small_rgb = image.resize(small_size, Image.Resampling.LANCZOS).convert("RGB")
        bgr = cv2.cvtColor(np.array(small_rgb), cv2.COLOR_RGB2BGR)

        small_face_mask = face_mask.resize(small_size, Image.Resampling.NEAREST)
        small_accessory_mask = accessory_mask.resize(small_size, Image.Resampling.NEAREST)
        small_fallback_hair = fallback_hair_mask.resize(small_size, Image.Resampling.NEAREST)

        mask = np.full((small_size[1], small_size[0]), cv2.GC_BGD, dtype=np.uint8)

        head_rect = (
            int(small_size[0] * 0.12),
            int(small_size[1] * 0.02),
            int(small_size[0] * 0.88),
            int(small_size[1] * 0.70),
        )
        hx1, hy1, hx2, hy2 = head_rect
        mask[hy1:hy2, hx1:hx2] = cv2.GC_PR_FGD

        if detection is not None:
            pts = np.array(detection.points * scale, dtype=np.int32)
            jaw = pts[0:17]
            brow_left = pts[17:22]
            brow_right = pts[22:27]
            face_poly = np.vstack([jaw, brow_right[::-1], brow_left[::-1]])
            cv2.fillConvexPoly(mask, face_poly, cv2.GC_BGD)
        else:
            face_np = np.array(small_face_mask, dtype=np.uint8)
            mask[face_np > 0] = cv2.GC_BGD

        acc_np = np.array(small_accessory_mask, dtype=np.uint8)
        mask[acc_np > 0] = cv2.GC_BGD

        fallback_np = np.array(small_fallback_hair, dtype=np.uint8)
        strong_seed = cv2.erode((fallback_np > 0).astype("uint8") * 255, np.ones((5, 5), np.uint8), iterations=1)
        mask[strong_seed > 0] = cv2.GC_FGD

        side_margin = max(1, int(small_size[0] * 0.06))
        lower_start = max(0, int(small_size[1] * 0.72))
        mask[lower_start:, :] = cv2.GC_BGD
        mask[:, :side_margin] = cv2.GC_BGD
        mask[:, small_size[0] - side_margin :] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
            small_result = Image.fromarray(fg, mode="L")
            small_result = ImageChops.subtract(
                small_result,
                reference_parser_service._dilate_mask(small_face_mask, 2),
            )
            small_result = ImageChops.subtract(
                small_result,
                reference_parser_service._dilate_mask(small_accessory_mask, 2),
            )
            refined_small = self._binarize(small_result)
            refined_full = refined_small.resize(image.size, Image.Resampling.NEAREST)
            merged = ImageChops.lighter(
                self._refine_hair_edit_mask(fallback_hair_mask, face_mask, accessory_mask, geometry),
                refined_full,
            )
            return self._refine_hair_edit_mask(merged, face_mask, accessory_mask, geometry)
        except Exception:
            return self._refine_hair_edit_mask(fallback_hair_mask, face_mask, accessory_mask, geometry)

    def _refine_hair_edit_mask(
        self,
        hair_mask: Image.Image,
        face_mask: Image.Image,
        accessory_mask: Image.Image,
        geometry,
    ) -> Image.Image:
        head_rect = reference_parser_service._rect_from_geom(geometry, -1.08, -1.08, 1.08, 0.62)
        head_limit = reference_parser_service._rect_mask(hair_mask.size, head_rect)
        refined = ImageChops.multiply(hair_mask, head_limit)
        refined = ImageChops.subtract(refined, reference_parser_service._dilate_mask(face_mask, 2))
        refined = ImageChops.subtract(refined, reference_parser_service._dilate_mask(accessory_mask, 2))
        refined = reference_parser_service._erode_mask(refined, 1)
        return self._binarize(refined)

    def _build_makeup_mask(
        self,
        masks: dict[str, Image.Image],
        face_mask: Image.Image,
        hair_mask: Image.Image,
        accessory_mask: Image.Image,
    ) -> Image.Image:
        makeup_regions = [
            masks["brow_left"],
            masks["brow_right"],
            masks["eye_band"],
            masks["liner_left"],
            masks["liner_right"],
            masks["under_eye"],
            masks["blush_left"],
            masks["blush_right"],
            masks["lips"],
            masks["nose_bridge"],
            masks["forehead_highlight"],
            masks["nose_tip"],
            masks["contour_left"],
            masks["contour_right"],
        ]
        merged = self._union_masks(makeup_regions)
        merged = ImageChops.multiply(merged, reference_parser_service._dilate_mask(face_mask, 1))
        merged = reference_parser_service._dilate_mask(merged, 1)
        merged = ImageChops.subtract(merged, reference_parser_service._dilate_mask(hair_mask, 2))
        merged = ImageChops.subtract(merged, reference_parser_service._dilate_mask(accessory_mask, 2))
        return self._binarize(merged)

    def _build_face_lock_mask(
        self,
        face_mask: Image.Image,
        hair_mask: Image.Image,
        makeup_mask: Image.Image,
        accessory_mask: Image.Image,
    ) -> Image.Image:
        base_face = reference_parser_service._dilate_mask(face_mask, 2)
        editable = self._union_masks([hair_mask, makeup_mask])
        locked = ImageChops.subtract(base_face, reference_parser_service._dilate_mask(editable, 2))
        locked = self._union_masks([locked, accessory_mask])
        return self._binarize(locked)

    def _build_landmark_component_masks(self, size: tuple[int, int], geometry, detection) -> dict[str, Image.Image]:
        pts = detection.points
        jaw = pts[0:17]
        brow_left = pts[17:22]
        brow_right = pts[22:27]
        nose = pts[27:36]
        eye_left = pts[36:42]
        eye_right = pts[42:48]
        mouth_outer = pts[48:60]
        mouth_inner = pts[60:68]
        face_poly = np.vstack([jaw, brow_right[::-1], brow_left[::-1]])

        return {
            "face": reference_parser_service._polygon_mask(size, face_poly, blur_radius=1),
            "brow_left": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._expand_poly(brow_left, 1.20, 1.35),
                blur_radius=1,
            ),
            "brow_right": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._expand_poly(brow_right, 1.20, 1.35),
                blur_radius=1,
            ),
            "eye_band": reference_parser_service._polygon_mask(
                size,
                np.vstack(
                    [
                        reference_parser_service._expand_poly(eye_left, 1.18, 1.25),
                        reference_parser_service._expand_poly(eye_right, 1.18, 1.25),
                    ]
                ),
                blur_radius=1,
            ),
            "liner_left": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._expand_poly(eye_left, 1.08, 1.10),
                blur_radius=1,
            ),
            "liner_right": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._expand_poly(eye_right, 1.08, 1.10),
                blur_radius=1,
            ),
            "under_eye": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._under_eye_poly(eye_left, eye_right),
                blur_radius=1,
            ),
            "blush_left": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._cheek_poly(jaw, nose, side="left"),
                blur_radius=2,
            ),
            "blush_right": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._cheek_poly(jaw, nose, side="right"),
                blur_radius=2,
            ),
            "lips": reference_parser_service._polygon_mask(
                size,
                mouth_outer,
                holes=[mouth_inner],
                blur_radius=1,
            ),
            "nose_bridge": reference_parser_service._polygon_mask(size, nose[0:4], blur_radius=1),
            "forehead_highlight": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._forehead_poly(face_poly, geometry),
                blur_radius=2,
            ),
            "nose_tip": reference_parser_service._polygon_mask(size, nose[3:6], blur_radius=1),
            "contour_left": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._contour_poly(jaw, side="left"),
                blur_radius=2,
            ),
            "contour_right": reference_parser_service._polygon_mask(
                size,
                reference_parser_service._contour_poly(jaw, side="right"),
                blur_radius=2,
            ),
        }

    def _union_masks(self, masks: list[Image.Image]) -> Image.Image:
        result = Image.new("L", masks[0].size, 0)
        for mask in masks:
            result = ImageChops.lighter(result, mask.convert("L"))
        return result

    def _binarize(self, mask: Image.Image, threshold: int = 24) -> Image.Image:
        return mask.convert("L").point(lambda value: 255 if value >= threshold else 0)

    def _mask_fill_ratio(self, mask: Image.Image) -> float:
        values = np.array(mask.convert("L"))
        return float((values > 0).mean()) if values.size else 0.0

    def _pseudo_identity_embedding(self, face_lock_mask: Image.Image) -> list[float]:
        arr = np.array(face_lock_mask.convert("L"), dtype=np.float32) / 255.0
        if arr.size == 0:
            return []
        small = np.array(Image.fromarray((arr * 255).astype("uint8")).resize((8, 8), Image.Resampling.BILINEAR))
        normalized = (small.astype(np.float32).flatten() / 255.0) - 0.5
        return [round(float(value), 4) for value in normalized[:32]]


preprocess_service = PreprocessService()
