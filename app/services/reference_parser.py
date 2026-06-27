from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat

try:
    import cv2
except Exception:
    cv2 = None

from app.models.pipeline import (
    AegyoSalFeatures,
    BangsFeatures,
    BaseMakeupFeatures,
    BlushFeatures,
    ColorFeature,
    ContourFeatures,
    EyebrowFeatures,
    EyelashesFeatures,
    EyelinerFeatures,
    EyeshadowFeatures,
    HairFeatures,
    HairSideLocks,
    HighlightFeatures,
    LipFeatures,
    MakeupFeatures,
    RegionMaskSet,
    ReferenceParseResult,
    TextureFeatures,
)
from app.utils.images import load_image_bytes


@dataclass
class RGB:
    r: float
    g: float
    b: float


@dataclass
class FaceGeometry:
    cx: float
    cy: float
    fw: float
    fh: float
    image_width: int
    image_height: int


@dataclass
class LandmarkDetection:
    face_rect: tuple[int, int, int, int]
    points: np.ndarray


@dataclass
class ParsedRegions:
    region_masks: RegionMaskSet
    left_strip: Image.Image
    right_strip: Image.Image
    upper_band: Image.Image
    lower_band: Image.Image
    crown_band: Image.Image
    side_band: Image.Image
    left_top: Image.Image
    right_top: Image.Image
    right_mid: Image.Image
    forehead_band: Image.Image
    center_forehead: Image.Image
    left_forehead: Image.Image
    right_forehead: Image.Image
    face_center: Image.Image
    left_cheek: Image.Image
    right_cheek: Image.Image
    lips: Image.Image
    left_brow: Image.Image
    right_brow: Image.Image
    eye_band: Image.Image
    liner_left: Image.Image
    liner_right: Image.Image
    under_eye: Image.Image
    nose_bridge: Image.Image
    forehead_highlight: Image.Image
    nose_tip: Image.Image
    top_bg: Image.Image
    side_bg: Image.Image


class ReferenceParserService:
    def __init__(self) -> None:
        self._facemark = None
        self._facemark_load_attempted = False

    def run(self, reference_image: str) -> ReferenceParseResult:
        image = Image.open(io.BytesIO(load_image_bytes(reference_image))).convert("RGB")
        regions = self._extract_regions(image)
        hair = self._parse_hair_structure(image, regions)
        bangs = self._parse_bangs_structure(regions, hair)
        hair = self._refine_hair_structure(hair, bangs, regions)
        makeup = self._parse_makeup_attributes(image, regions)
        texture = self._parse_texture_attributes(regions)
        confidence = self._aggregate_confidence(hair, bangs, makeup, texture)
        style_caption, consistency_flags, confidence_overrides = self._merge_vlm_summary(
            hair=hair,
            bangs=bangs,
            makeup=makeup,
            texture=texture,
            parse_confidence=confidence,
        )

        return ReferenceParseResult(
            region_masks=regions.region_masks,
            hair_features=hair,
            bangs=bangs,
            makeup_features=makeup,
            texture_features=texture,
            style_caption=style_caption,
            consistency_flags=consistency_flags,
            field_confidence_overrides=confidence_overrides,
            negative_constraints=[
                "do not inherit reference identity",
                "do not inherit reference clothing",
                "do not inherit reference background",
            ],
            normalized_prompt_tokens=self._build_tokens(hair, bangs, makeup, texture),
            parse_confidence=confidence,
        )

    def _extract_regions(self, image: Image.Image) -> ParsedRegions:
        geometry = self._estimate_face_geometry(image)
        masks = self._build_region_masks(image, geometry)

        region_masks = RegionMaskSet(
            hair="derived://region/hair",
            bangs="derived://region/bangs",
            hairline="derived://region/hairline",
            brow_left="derived://region/brow_left",
            brow_right="derived://region/brow_right",
            upper_eyelid_left="derived://region/upper_eyelid_left",
            upper_eyelid_right="derived://region/upper_eyelid_right",
            lower_eyelid_left="derived://region/lower_eyelid_left",
            lower_eyelid_right="derived://region/lower_eyelid_right",
            eyelashes_upper="derived://region/eyelashes_upper",
            eyelashes_lower="derived://region/eyelashes_lower",
            lips="derived://region/lips",
            blush_left="derived://region/blush_left",
            blush_right="derived://region/blush_right",
            nose_highlight="derived://region/nose_highlight",
            contour_left="derived://region/contour_left",
            contour_right="derived://region/contour_right",
        )

        left_strip_rect = self._rect_from_geom(geometry, -1.55, -0.20, -0.72, 1.10)
        right_strip_rect = self._rect_from_geom(geometry, 0.72, -0.20, 1.55, 1.10)
        upper_band_rect = self._rect_from_geom(geometry, -1.15, -1.25, 1.15, -0.52)
        lower_band_rect = self._rect_from_geom(geometry, -1.05, 0.62, 1.05, 1.52)
        crown_band_rect = self._rect_from_geom(geometry, -0.70, -1.20, 0.70, -0.62)
        side_band_rect = self._rect_from_geom(geometry, -1.20, -0.32, 1.20, 0.38)
        left_top_rect = self._rect_from_geom(geometry, -0.88, -1.05, -0.10, -0.10)
        right_top_rect = self._rect_from_geom(geometry, 0.10, -1.05, 0.88, -0.10)
        right_mid_rect = self._rect_from_geom(geometry, 0.45, -0.18, 1.25, 0.98)
        forehead_rect = self._rect_from_geom(geometry, -0.66, -0.88, 0.66, -0.28)
        center_forehead_rect = self._rect_from_geom(geometry, -0.20, -0.82, 0.20, -0.30)
        left_forehead_rect = self._rect_from_geom(geometry, -0.66, -0.82, -0.12, -0.26)
        right_forehead_rect = self._rect_from_geom(geometry, 0.12, -0.82, 0.66, -0.26)
        face_center_rect = self._rect_from_geom(geometry, -0.48, -0.28, 0.48, 0.52)
        left_cheek_rect = self._rect_from_geom(geometry, -0.82, 0.06, -0.26, 0.58)
        right_cheek_rect = self._rect_from_geom(geometry, 0.26, 0.06, 0.82, 0.58)
        lips_rect = self._rect_from_geom(geometry, -0.30, 0.46, 0.30, 0.88)
        left_brow_rect = self._rect_from_geom(geometry, -0.70, -0.26, -0.18, -0.02)
        right_brow_rect = self._rect_from_geom(geometry, 0.18, -0.26, 0.70, -0.02)
        eye_band_rect = self._rect_from_geom(geometry, -0.74, -0.04, 0.74, 0.34)
        liner_left_rect = self._rect_from_geom(geometry, -0.68, 0.02, -0.16, 0.20)
        liner_right_rect = self._rect_from_geom(geometry, 0.16, 0.02, 0.68, 0.20)
        under_eye_rect = self._rect_from_geom(geometry, -0.62, 0.20, 0.62, 0.44)
        nose_bridge_rect = self._rect_from_geom(geometry, -0.11, -0.02, 0.11, 0.52)
        forehead_highlight_rect = self._rect_from_geom(geometry, -0.34, -0.96, 0.34, -0.42)
        nose_tip_rect = self._rect_from_geom(geometry, -0.12, 0.36, 0.12, 0.58)
        top_bg_rect = self._clamp_rect((0, 0, image.size[0], max(1, int(image.size[1] * 0.18))), image.size)
        side_bg_rect = self._clamp_rect(
            (0, int(image.size[1] * 0.16), max(1, int(image.size[0] * 0.12)), int(image.size[1] * 0.90)),
            image.size,
        )

        return ParsedRegions(
            region_masks=region_masks,
            left_strip=self._extract_masked_region(image, masks["hair"], left_strip_rect),
            right_strip=self._extract_masked_region(image, masks["hair"], right_strip_rect),
            upper_band=self._extract_masked_region(image, masks["hair"], upper_band_rect),
            lower_band=self._extract_masked_region(image, masks["hair"], lower_band_rect),
            crown_band=self._extract_masked_region(image, masks["hair"], crown_band_rect),
            side_band=self._extract_masked_region(image, masks["hair"], side_band_rect),
            left_top=self._extract_masked_region(image, masks["hair"], left_top_rect),
            right_top=self._extract_masked_region(image, masks["hair"], right_top_rect),
            right_mid=self._extract_masked_region(image, masks["hair"], right_mid_rect),
            forehead_band=self._extract_masked_region(image, masks["bangs"], forehead_rect),
            center_forehead=self._extract_masked_region(image, masks["bangs"], center_forehead_rect),
            left_forehead=self._extract_masked_region(image, masks["bangs"], left_forehead_rect),
            right_forehead=self._extract_masked_region(image, masks["bangs"], right_forehead_rect),
            face_center=self._extract_masked_region(image, masks["face"], face_center_rect),
            left_cheek=self._extract_masked_region(image, masks["blush_left"], left_cheek_rect),
            right_cheek=self._extract_masked_region(image, masks["blush_right"], right_cheek_rect),
            lips=self._extract_masked_region(image, masks["lips"], lips_rect),
            left_brow=self._extract_masked_region(image, masks["brow_left"], left_brow_rect),
            right_brow=self._extract_masked_region(image, masks["brow_right"], right_brow_rect),
            eye_band=self._extract_masked_region(image, masks["eye_band"], eye_band_rect),
            liner_left=self._extract_masked_region(image, masks["liner_left"], liner_left_rect),
            liner_right=self._extract_masked_region(image, masks["liner_right"], liner_right_rect),
            under_eye=self._extract_masked_region(image, masks["under_eye"], under_eye_rect),
            nose_bridge=self._extract_masked_region(image, masks["nose_bridge"], nose_bridge_rect),
            forehead_highlight=self._extract_masked_region(image, masks["forehead_highlight"], forehead_highlight_rect),
            nose_tip=self._extract_masked_region(image, masks["nose_tip"], nose_tip_rect),
            top_bg=self._extract_masked_region(image, masks["background_top"], top_bg_rect),
            side_bg=self._extract_masked_region(image, masks["background_side"], side_bg_rect),
        )

    def _build_region_masks_heuristic(self, image: Image.Image, geometry: FaceGeometry) -> dict[str, Image.Image]:
        detection = self._detect_landmarks(image, geometry)
        if detection is not None:
            try:
                return self._build_region_masks_from_landmarks(image, geometry, detection)
            except Exception:
                pass
        return self._build_region_masks_heuristic(image, geometry)

    def _parse_hair_structure(self, image: Image.Image, regions: ParsedRegions) -> HairFeatures:
        left_dark = self._dark_ratio(regions.left_strip)
        right_dark = self._dark_ratio(regions.right_strip)
        upper_dark = self._dark_ratio(regions.upper_band)
        lower_dark = self._dark_ratio(regions.lower_band)
        crown_dark = self._dark_ratio(regions.crown_band)

        style = "down"
        updo_type = "none"
        style_conf = 0.74
        if upper_dark > 0.58 and lower_dark < 0.14 and left_dark < 0.16 and right_dark < 0.16:
            style = "updo"
            updo_type = "bun_or_ponytail"
            style_conf = 0.62
        elif upper_dark > 0.42 and lower_dark < 0.18 and max(left_dark, right_dark) < 0.20:
            style = "half_up"
            updo_type = "half_up_clip"
            style_conf = 0.54

        length = "long"
        if lower_dark < 0.14 and max(left_dark, right_dark) < 0.18:
            length = "medium"
        if lower_dark < 0.08 and max(left_dark, right_dark) < 0.12:
            length = "short"

        left_top_dark = self._dark_ratio(regions.left_top)
        right_top_dark = self._dark_ratio(regions.right_top)
        top_delta = left_top_dark - right_top_dark
        if abs(top_delta) < 0.035:
            parting = "middle"
            parting_conf = 0.58
        elif top_delta > 0.08:
            parting = "side_3_7"
            parting_conf = 0.82
        elif top_delta > 0.04:
            parting = "side_4_6"
            parting_conf = 0.72
        elif top_delta < -0.08:
            parting = "side_7_3"
            parting_conf = 0.82
        else:
            parting = "side_6_4"
            parting_conf = 0.72

        texture = self._classify_hair_texture(regions.right_mid)
        volume_crown = min(0.95, max(0.10, crown_dark * 1.45))
        volume_side = min(0.95, max(0.10, self._dark_ratio(regions.side_band) * 1.05))
        hairline_exposure = round(
            max(0.10, min(0.90, 1.0 - self._dark_ratio(regions.center_forehead))),
            2,
        )

        hair_sample = self._pick_hair_color_sample(image, regions, left_dark, right_dark)
        hair_rgb = self._median_rgb_non_background(hair_sample)
        hair_color = ColorFeature(
            label=self._classify_hair_color(hair_rgb),
            hex=self._rgb_to_hex(hair_rgb),
            confidence=0.78,
        )

        sleekness = round(max(0.10, min(0.95, 1.0 - volume_side * 0.75)), 2)

        slicked_back_candidate = (
            hairline_exposure >= 0.80
            and volume_side <= 0.24
            and lower_dark <= 0.14
            and sleekness >= 0.80
        )
        if slicked_back_candidate:
            style = "updo"
            updo_type = "tight_bun"
            style_conf = 0.86
            parting = "none_or_natural_back"
            texture = "straight_sleek"
        elif hairline_exposure >= 0.76 and volume_side <= 0.24 and lower_dark <= 0.18 and upper_dark >= 0.24:
            style = "updo"
            updo_type = "bun_or_ponytail"
            style_conf = 0.74
            parting = "none_or_natural_back"

        side_lock_exists = style == "down" and (left_dark > 0.12 or right_dark > 0.12)
        side_lock_length = "medium" if side_lock_exists and length != "short" else "none"
        side_lock_intensity = 0.52 if side_lock_exists else 0.0
        if style == "updo":
            hairline_exposure = max(hairline_exposure, 0.58)

        primary_style = style
        secondary_style = updo_type
        finish = "sleek" if texture in {"straight", "straight_sleek"} else "natural"
        if slicked_back_candidate:
            primary_style = "slicked_back_updo"
            secondary_style = "tight_bun"
            finish = "sleek"
        elif style == "updo" and sleekness > 0.62:
            primary_style = "slicked_back_updo"
            secondary_style = "tight_bun"
            finish = "sleek"
        elif style == "half_up":
            primary_style = "half_up"
        elif texture in {"soft_wave", "wavy"}:
            primary_style = "down"
        gloss = round(max(0.05, min(0.90, 0.25 + self._highlight_ratio(regions.right_mid) * 1.1)), 2)

        return HairFeatures(
            style=style if style_conf >= 0.5 else "unknown",
            updo_type=updo_type,
            length=length,
            parting=parting if parting_conf >= 0.60 else "unclear",
            texture=texture,
            color=hair_color,
            volume_crown=round(volume_crown, 2),
            volume_side=round(volume_side, 2),
            hairline_exposure=hairline_exposure,
            side_locks=HairSideLocks(
                exists=side_lock_exists,
                length=side_lock_length,
                curl=0.38 if texture in {"soft_wave", "wavy"} else 0.14,
                intensity=round(side_lock_intensity, 2),
            ),
            primary_style=primary_style,
            secondary_style=secondary_style,
            finish=finish,
            gloss=gloss,
            sleekness=sleekness,
            confidence=round(style_conf, 2),
        )

    def _parse_bangs_structure(self, regions: ParsedRegions, hair: HairFeatures) -> BangsFeatures:
        forehead_dark = self._dark_ratio(regions.forehead_band)
        center_dark = self._dark_ratio(regions.center_forehead)
        left_dark = self._dark_ratio(regions.left_forehead)
        right_dark = self._dark_ratio(regions.right_forehead)
        exists = forehead_dark > 0.12 or center_dark > 0.08 or abs(left_dark - right_dark) > 0.10

        if not exists:
            return BangsFeatures(
                exists=False,
                type="none",
                density=0.0,
                length="none",
                curve="none",
                gap_ratio=0.0,
                confidence=0.50,
            )

        asymmetry = abs(left_dark - right_dark)
        gap_ratio = round(max(0.0, min(1.0, 1.0 - center_dark / max(forehead_dark, 0.001))), 2)
        if hair.parting.startswith("side") and (gap_ratio > 0.08 or asymmetry > 0.05):
            bang_type = "airy_side_bangs"
            density = 0.34
        elif asymmetry > 0.10:
            bang_type = "side_swept_bangs"
            density = 0.44
        elif gap_ratio > 0.35:
            bang_type = "see_through_bangs"
            density = 0.32
        else:
            bang_type = "soft_bangs"
            density = 0.50

        return BangsFeatures(
            exists=True,
            type=bang_type,
            density=round(density, 2),
            length="brow_to_eye",
            curve="soft",
            gap_ratio=gap_ratio,
            confidence=0.72,
        )

    def _refine_hair_structure(
        self,
        hair: HairFeatures,
        bangs: BangsFeatures,
        regions: ParsedRegions,
    ) -> HairFeatures:
        if (
            not bangs.exists
            and hair.hairline_exposure >= 0.82
            and hair.volume_side <= 0.24
            and hair.sleekness >= 0.80
        ):
            hair.style = "updo"
            hair.updo_type = "tight_bun"
            hair.primary_style = "slicked_back_updo"
            hair.secondary_style = "tight_bun"
            hair.parting = "none_or_natural_back"
            hair.texture = "straight_sleek"
            hair.finish = "sleek"
            hair.confidence = max(hair.confidence, 0.86)
            hair.side_locks.exists = False
            hair.side_locks.length = "none"
            hair.side_locks.intensity = 0.0
            hair.side_locks.curl = 0.14
        elif (
            not bangs.exists
            and hair.hairline_exposure >= 0.72
            and hair.volume_side <= 0.28
            and self._dark_ratio(regions.lower_band) <= 0.18
        ):
            hair.style = "updo"
            hair.updo_type = "bun_or_ponytail"
            hair.primary_style = "updo"
            hair.secondary_style = "bun_or_ponytail"
            hair.parting = "none_or_natural_back"
            hair.confidence = max(hair.confidence, 0.76)
        return hair

    def _parse_makeup_attributes(self, image: Image.Image, regions: ParsedRegions) -> MakeupFeatures:
        face_rgb = self._median_rgb(regions.face_center)
        cheek_rgb = self._median_rgb(self._blend_regions(regions.left_cheek, regions.right_cheek))
        lip_rgb = self._median_rgb(regions.lips)
        brow_mix = self._blend_regions(regions.left_brow, regions.right_brow)
        brow_rgb = self._median_rgb(brow_mix)
        eye_rgb = self._median_rgb(regions.eye_band)
        under_eye_rgb = self._median_rgb(regions.under_eye)
        nose_rgb = self._median_rgb(regions.nose_bridge)

        face_brightness = self._brightness(face_rgb)
        face_glow = self._highlight_ratio(regions.face_center)
        forehead_glow = self._highlight_ratio(regions.forehead_highlight)
        nose_tip_glow = self._highlight_ratio(regions.nose_tip)
        face_evenness = min(0.96, max(0.20, 1.0 - self._channel_std(regions.face_center) / 95.0))
        lip_saturation = self._saturation(lip_rgb)
        blush_saturation = self._saturation(cheek_rgb)
        eye_saturation = self._saturation(eye_rgb)
        brow_darkness = 1.0 - self._brightness(brow_rgb)
        brow_std = self._channel_std(brow_mix)
        liner_density = max(self._dark_ratio(regions.liner_left), self._dark_ratio(regions.liner_right))
        under_eye_std = self._channel_std(regions.under_eye)
        lip_std = self._channel_std(regions.lips)

        base_finish = self._classify_skin_finish(regions.face_center)
        if (face_glow > 0.16 or nose_tip_glow > 0.55 or forehead_glow > 0.10) and face_evenness > 0.45:
            base_finish = "semi_glowy"
        base_intensity = min(0.92, max(0.36, 0.45 + face_evenness * 0.30))
        base_coverage = min(0.94, max(0.30, 0.35 + face_evenness * 0.45))
        brightness_shift = round(face_brightness - 0.62, 2)
        powderiness = round(max(0.04, 0.48 - face_glow * 0.55), 2)

        blush_color = self._classify_makeup_color(cheek_rgb, role="blush")
        lip_color = self._classify_makeup_color(lip_rgb, role="lips")
        eye_color = self._classify_makeup_color(eye_rgb, role="eyes")
        highlight_color = self._classify_makeup_color(under_eye_rgb, role="highlight")
        contour_color = self._classify_makeup_color(nose_rgb, role="contour")
        brow_color = self._classify_makeup_color(brow_rgb, role="brow")

        return MakeupFeatures(
            base_makeup=BaseMakeupFeatures(
                finish=base_finish,
                coverage=round(base_coverage, 2),
                brightness_shift=brightness_shift,
                evenness=round(face_evenness, 2),
                glow=round(face_glow, 2),
                powderiness=powderiness,
                intensity=round(base_intensity, 2),
                concealer_coverage=round(base_coverage, 2),
                brightness_level=round(face_brightness, 2),
                finish_confidence=0.72,
            ),
            blush=BlushFeatures(
                color=blush_color,
                placement="mid_cheek",
                shape="soft_oval",
                range=0.34,
                intensity=round(max(0.18, min(0.88, blush_saturation + 0.12)), 2),
                confidence=0.68,
            ),
            contour=ContourFeatures(
                color=contour_color,
                nose_contour=round(max(0.08, min(0.78, (1.0 - self._brightness(nose_rgb)) * 0.8)), 2),
                cheek_contour=round(max(0.06, min(0.74, (1.0 - self._brightness(cheek_rgb)) * 0.55)), 2),
                jaw_contour=0.18,
                intensity=round(max(0.10, min(0.68, (1.0 - self._brightness(nose_rgb)) * 0.65)), 2),
                confidence=0.62,
            ),
            highlight=HighlightFeatures(
                color=highlight_color,
                nose_highlight=round(max(0.10, min(0.78, face_glow + 0.12)), 2),
                cheek_highlight=round(max(0.10, min(0.76, face_glow + 0.08)), 2),
                under_eye_highlight=round(max(0.10, min(0.72, 0.26 + under_eye_std / 80.0)), 2),
                forehead_highlight=round(max(0.08, min(0.72, forehead_glow + 0.10)), 2),
                nose_tip_highlight=round(max(0.08, min(0.80, nose_tip_glow + 0.08)), 2),
                intensity=round(max(0.10, min(0.80, face_glow + 0.10)), 2),
                confidence=0.66,
            ),
            eyebrow=EyebrowFeatures(
                shape=(
                    "straight_soft_arch"
                    if brow_std > 26 and brow_darkness < 0.58
                    else ("soft_arch" if brow_darkness > 0.26 else "straight_soft")
                ),
                color=brow_color,
                thickness=round(max(0.16, min(0.84, brow_darkness)), 2),
                arch=round(max(0.10, min(0.72, 0.30 + brow_std / 80.0)), 2),
                hair_texture=round(max(0.10, min(0.76, brow_std / 60.0)), 2),
                intensity=round(max(0.18, min(0.90, brow_darkness + 0.14)), 2),
                confidence=0.74,
            ),
            eyeliner=EyelinerFeatures(
                color="black" if liner_density > 0.20 else "soft_brown",
                style=(
                    "thin_lifted"
                    if 0.24 <= liner_density <= 0.40
                    else ("micro_wing" if liner_density > 0.40 else "inner_defined")
                ),
                length=round(max(0.12, min(0.88, 0.24 + liner_density * 1.4)), 2),
                thickness=round(max(0.08, min(0.72, 0.12 + liner_density * 0.9)), 2),
                tail_direction="slightly_up",
                intensity=round(max(0.10, min(0.92, 0.18 + liner_density * 1.6)), 2),
                confidence=0.75,
            ),
            eyeshadow=EyeshadowFeatures(
                main_color=eye_color,
                secondary_color=self._classify_secondary_eye_color(eye_rgb),
                placement="upper_lid_outer_corner",
                gradient="soft_gradient",
                finish="satin" if eye_saturation > 0.18 else "matte",
                intensity=round(max(0.12, min(0.82, eye_saturation + 0.10)), 2),
                confidence=0.63,
            ),
            eyelashes=EyelashesFeatures(
                upper_density=round(max(0.16, min(0.92, liner_density * 2.1 + 0.10)), 2),
                lower_density=round(max(0.05, min(0.55, liner_density * 0.8)), 2),
                length=round(max(0.16, min(0.88, liner_density * 1.8 + 0.14)), 2),
                curl=round(max(0.12, min(0.84, 0.22 + self._channel_std(regions.eye_band) / 65.0)), 2),
                cluster_style="natural_separated",
                intensity=round(max(0.14, min(0.88, liner_density * 1.6 + 0.12)), 2),
                confidence=0.67,
            ),
            aegyo_sal=AegyoSalFeatures(
                exists=under_eye_std > 10,
                highlight_color=highlight_color,
                shadow_color=contour_color,
                shape="soft_parallel",
                intensity=round(max(0.06, min(0.68, under_eye_std / 28.0)), 2),
                confidence=0.58,
            ),
            lips=LipFeatures(
                color=lip_color,
                shape="defined_full" if lip_saturation > 0.55 else "soft_full",
                edge_blur=0.46,
                gloss=round(max(0.06, min(0.88, face_glow * 0.70 + 0.08)), 2),
                saturation=round(lip_saturation, 2),
                intensity=round(max(0.22, min(0.92, lip_saturation + 0.16)), 2),
                edge_definition=round(max(0.10, min(0.92, 0.30 + lip_saturation * 0.65 + lip_std / 220.0)), 2),
                cupid_bow_definition=round(max(0.18, min(0.88, 0.34 + lip_saturation * 0.42)), 2),
                bite_effect=round(max(0.04, min(0.46, 0.22 - lip_saturation * 0.12)), 2),
                confidence=0.72,
            ),
        )

    def _parse_texture_attributes(self, regions: ParsedRegions) -> TextureFeatures:
        bg_mix = self._blend_regions(regions.top_bg, regions.side_bg)
        bg_std = self._channel_std(bg_mix)
        bg_brightness = self._brightness(self._median_rgb(bg_mix))

        if bg_std < 10 and bg_brightness > 0.82:
            photo_style = "clean_studio_portrait"
        elif bg_std < 14:
            photo_style = "soft_portrait"
        else:
            photo_style = "natural_photo"

        if bg_brightness > 0.82:
            overall_vibe = "clean_commute"
        elif bg_std < 18:
            overall_vibe = "soft_lifestyle"
        else:
            overall_vibe = "daily_photo"

        return TextureFeatures(
            skin_finish=self._classify_skin_finish(regions.face_center),
            photo_style=photo_style,
            overall_vibe=overall_vibe,
            confidence=0.70,
        )

    def _merge_vlm_summary(
        self,
        hair: HairFeatures,
        bangs: BangsFeatures,
        makeup: MakeupFeatures,
        texture: TextureFeatures,
        parse_confidence: float,
    ) -> tuple[str, list[str], dict[str, float]]:
        consistency_flags: list[str] = []
        if hair.primary_style == "slicked_back_updo" and bangs.exists:
            consistency_flags.append("bangs_and_slicked_back_conflict")
        if hair.hairline_exposure > 0.82 and bangs.exists:
            consistency_flags.append("high_hairline_exposure_with_bangs")

        bang_text = f"with {bangs.type}" if bangs.exists else "with no bangs"
        style_caption = (
            f"{hair.primary_style or hair.style} {hair.texture} hair, "
            f"{bang_text}, {makeup.base_makeup.finish} skin, "
            f"{makeup.eyeshadow.main_color} eyes, {makeup.lips.color} lips, "
            f"{texture.overall_vibe} vibe"
        )
        confidence_overrides = {
            "hair.primary_style": round(hair.confidence, 2),
            "hair.parting": 0.72 if hair.parting not in {"unknown", "unclear"} else 0.45,
            "bangs.type": round(bangs.confidence, 2),
            "makeup.base.finish": makeup.base_makeup.finish_confidence,
            "makeup.lips.color": makeup.lips.confidence,
            "texture.photo_style": texture.confidence,
            "reference.parse_confidence": parse_confidence,
        }
        return style_caption, consistency_flags, confidence_overrides

    def _aggregate_confidence(
        self,
        hair: HairFeatures,
        bangs: BangsFeatures,
        makeup: MakeupFeatures,
        texture: TextureFeatures,
    ) -> float:
        score = 0.50
        if hair.style != "unknown":
            score += 0.10
        if hair.parting not in {"unknown", "unclear"}:
            score += 0.08
        if hair.color.confidence >= 0.70:
            score += 0.08
        if bangs.exists:
            score += 0.06
        if makeup.lips.intensity >= 0.20:
            score += 0.05
        if makeup.eyeliner.intensity >= 0.20:
            score += 0.05
        if texture.photo_style != "unknown":
            score += 0.05
        return round(min(0.94, score), 2)

    def _build_tokens(
        self,
        hair: HairFeatures,
        bangs: BangsFeatures,
        makeup: MakeupFeatures,
        texture: TextureFeatures,
    ) -> list[str]:
        tokens = [hair.style, hair.length, hair.texture]
        if hair.updo_type != "none":
            tokens.append(hair.updo_type)
        if hair.parting not in {"unclear", "unknown"}:
            tokens.append(hair.parting)
        if bangs.exists:
            tokens.append(bangs.type)
        tokens.extend(
            [
                f"primary_style:{hair.primary_style}",
                f"hair_color:{hair.color.label}",
                f"base_finish:{makeup.base_makeup.finish}",
                f"blush:{makeup.blush.color}",
                f"lip:{makeup.lips.color}",
                f"eyeshadow:{makeup.eyeshadow.main_color}",
                f"eyeliner:{makeup.eyeliner.style}",
                texture.photo_style,
                texture.overall_vibe,
            ]
        )
        return tokens

    def _estimate_face_geometry(self, image: Image.Image) -> FaceGeometry:
        width, height = image.size
        search_rect = (
            int(width * 0.18),
            int(height * 0.10),
            int(width * 0.82),
            int(height * 0.88),
        )
        skin_pixels = self._find_skin_pixels(image, search_rect)
        if skin_pixels:
            xs = [p[0] for p in skin_pixels]
            ys = [p[1] for p in skin_pixels]
            x1 = min(xs)
            x2 = max(xs)
            y1 = min(ys)
            y2 = max(ys)
            cx = (x1 + x2) / 2.0
            cy = y1 + (y2 - y1) * 0.48
            fw = max(width * 0.20, (x2 - x1) * 0.92)
            fh = max(height * 0.24, (y2 - y1) * 0.72)
        else:
            cx = width * 0.50
            cy = height * 0.42
            fw = width * 0.34
            fh = height * 0.38
        return FaceGeometry(
            cx=cx,
            cy=cy,
            fw=min(width * 0.62, fw),
            fh=min(height * 0.62, fh),
            image_width=width,
            image_height=height,
        )

    def _build_region_masks(self, image: Image.Image, geometry: FaceGeometry) -> dict[str, Image.Image]:
        size = image.size
        face_mask = self._ellipse_mask(size, self._rect_from_geom(geometry, -0.62, -0.56, 0.62, 0.84))
        forehead_mask = self._ellipse_mask(size, self._rect_from_geom(geometry, -0.54, -0.78, 0.54, -0.18))
        left_brow_rect = self._rect_from_geom(geometry, -0.70, -0.24, -0.18, -0.02)
        right_brow_rect = self._rect_from_geom(geometry, 0.18, -0.24, 0.70, -0.02)
        lips_rect = self._rect_from_geom(geometry, -0.30, 0.46, 0.30, 0.86)
        left_eye_rect = self._rect_from_geom(geometry, -0.74, -0.02, -0.14, 0.28)
        right_eye_rect = self._rect_from_geom(geometry, 0.14, -0.02, 0.74, 0.28)
        eye_band_rect = self._rect_from_geom(geometry, -0.78, -0.06, 0.78, 0.34)
        under_eye_rect = self._rect_from_geom(geometry, -0.68, 0.18, 0.68, 0.46)
        nose_bridge_rect = self._rect_from_geom(geometry, -0.12, -0.04, 0.12, 0.52)
        forehead_highlight_rect = self._rect_from_geom(geometry, -0.34, -0.88, 0.34, -0.40)
        nose_tip_rect = self._rect_from_geom(geometry, -0.12, 0.36, 0.12, 0.58)
        hair_rect = self._rect_from_geom(geometry, -1.36, -1.34, 1.36, 1.18)
        left_cheek_rect = self._rect_from_geom(geometry, -0.86, 0.08, -0.24, 0.62)
        right_cheek_rect = self._rect_from_geom(geometry, 0.24, 0.08, 0.86, 0.62)
        contour_left_rect = self._rect_from_geom(geometry, -1.02, 0.00, -0.46, 0.72)
        contour_right_rect = self._rect_from_geom(geometry, 0.46, 0.00, 1.02, 0.72)
        top_bg_rect = self._clamp_rect((0, 0, size[0], max(1, int(size[1] * 0.18))), size)
        side_bg_rect = self._clamp_rect(
            (0, int(size[1] * 0.16), max(1, int(size[0] * 0.12)), int(size[1] * 0.90)),
            size,
        )

        hair_base = self._create_pixel_mask(
            image,
            hair_rect,
            predicate=lambda rgb, hsv, xy: (
                self._brightness(rgb) < 0.58
                or (hsv[1] > 0.18 and self._brightness(rgb) < 0.72)
            ),
        )
        hair_base = self._dilate_mask(hair_base, 4)
        hair_mask = ImageChops.subtract(hair_base, self._erode_mask(face_mask, 3))

        bangs_seed = ImageChops.multiply(hair_mask, forehead_mask)
        bangs_mask = self._dilate_mask(bangs_seed, 2)

        lips_mask = self._create_pixel_mask(
            image,
            lips_rect,
            predicate=lambda rgb, hsv, xy: (
                (rgb[0] - rgb[1]) > 6 and hsv[1] > 0.14
            ) or (hsv[0] < 0.08 and hsv[1] > 0.22),
        )
        lips_mask = self._merge_with_shape_hint(lips_mask, self._ellipse_mask(size, lips_rect), minimum_fill=0.28)

        left_brow_mask = self._create_pixel_mask(
            image,
            left_brow_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) < 0.45 or hsv[1] > 0.18,
        )
        right_brow_mask = self._create_pixel_mask(
            image,
            right_brow_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) < 0.45 or hsv[1] > 0.18,
        )
        left_brow_mask = self._merge_with_shape_hint(left_brow_mask, self._rect_mask(size, left_brow_rect), minimum_fill=0.22)
        right_brow_mask = self._merge_with_shape_hint(right_brow_mask, self._rect_mask(size, right_brow_rect), minimum_fill=0.22)

        eye_band_mask = self._create_pixel_mask(
            image,
            eye_band_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) < 0.62 or hsv[1] > 0.10,
        )
        eye_band_mask = self._merge_with_shape_hint(eye_band_mask, self._rect_mask(size, eye_band_rect), minimum_fill=0.25)

        liner_left_mask = self._create_pixel_mask(
            image,
            left_eye_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) < 0.36,
        )
        liner_right_mask = self._create_pixel_mask(
            image,
            right_eye_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) < 0.36,
        )
        liner_left_mask = self._merge_with_shape_hint(liner_left_mask, self._rect_mask(size, left_eye_rect), minimum_fill=0.16)
        liner_right_mask = self._merge_with_shape_hint(liner_right_mask, self._rect_mask(size, right_eye_rect), minimum_fill=0.16)

        under_eye_mask = self._create_pixel_mask(
            image,
            under_eye_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) > 0.55 or hsv[1] > 0.08,
        )
        under_eye_mask = self._merge_with_shape_hint(under_eye_mask, self._rect_mask(size, under_eye_rect), minimum_fill=0.24)

        blush_left_mask = self._create_pixel_mask(
            image,
            left_cheek_rect,
            predicate=lambda rgb, hsv, xy: hsv[1] > 0.12 and rgb[0] >= rgb[1] - 8,
        )
        blush_right_mask = self._create_pixel_mask(
            image,
            right_cheek_rect,
            predicate=lambda rgb, hsv, xy: hsv[1] > 0.12 and rgb[0] >= rgb[1] - 8,
        )
        blush_left_mask = self._merge_with_shape_hint(blush_left_mask, self._ellipse_mask(size, left_cheek_rect), minimum_fill=0.22)
        blush_right_mask = self._merge_with_shape_hint(blush_right_mask, self._ellipse_mask(size, right_cheek_rect), minimum_fill=0.22)

        contour_left_mask = self._create_pixel_mask(
            image,
            contour_left_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) < 0.58,
        )
        contour_right_mask = self._create_pixel_mask(
            image,
            contour_right_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) < 0.58,
        )
        contour_left_mask = self._merge_with_shape_hint(contour_left_mask, self._ellipse_mask(size, contour_left_rect), minimum_fill=0.18)
        contour_right_mask = self._merge_with_shape_hint(contour_right_mask, self._ellipse_mask(size, contour_right_rect), minimum_fill=0.18)

        nose_bridge_mask = self._create_pixel_mask(
            image,
            nose_bridge_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) > 0.42,
        )
        nose_bridge_mask = self._merge_with_shape_hint(nose_bridge_mask, self._rect_mask(size, nose_bridge_rect), minimum_fill=0.30)

        forehead_highlight_mask = self._create_pixel_mask(
            image,
            forehead_highlight_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) > 0.62,
        )
        forehead_highlight_mask = self._merge_with_shape_hint(
            forehead_highlight_mask,
            self._ellipse_mask(size, forehead_highlight_rect),
            minimum_fill=0.16,
        )

        nose_tip_mask = self._create_pixel_mask(
            image,
            nose_tip_rect,
            predicate=lambda rgb, hsv, xy: self._brightness(rgb) > 0.62,
        )
        nose_tip_mask = self._merge_with_shape_hint(nose_tip_mask, self._ellipse_mask(size, nose_tip_rect), minimum_fill=0.22)

        background_top = ImageChops.subtract(self._rect_mask(size, top_bg_rect), hair_mask)
        background_side = ImageChops.subtract(self._rect_mask(size, side_bg_rect), hair_mask)

        return {
            "face": self._merge_with_shape_hint(face_mask, face_mask, minimum_fill=0.40),
            "hair": self._merge_with_shape_hint(hair_mask, self._rect_mask(size, hair_rect), minimum_fill=0.20),
            "bangs": self._merge_with_shape_hint(bangs_mask, self._ellipse_mask(size, forehead_mask.getbbox() or forehead_rect), minimum_fill=0.10),
            "brow_left": left_brow_mask,
            "brow_right": right_brow_mask,
            "eye_band": eye_band_mask,
            "liner_left": liner_left_mask,
            "liner_right": liner_right_mask,
            "under_eye": under_eye_mask,
            "blush_left": blush_left_mask,
            "blush_right": blush_right_mask,
            "lips": lips_mask,
            "nose_bridge": nose_bridge_mask,
            "forehead_highlight": forehead_highlight_mask,
            "nose_tip": nose_tip_mask,
            "background_top": self._merge_with_shape_hint(background_top, self._rect_mask(size, top_bg_rect), minimum_fill=0.40),
            "background_side": self._merge_with_shape_hint(background_side, self._rect_mask(size, side_bg_rect), minimum_fill=0.40),
            "contour_left": contour_left_mask,
            "contour_right": contour_right_mask,
        }

    def _find_skin_pixels(self, image: Image.Image, rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
        pixels: list[tuple[int, int]] = []
        x1, y1, x2, y2 = rect
        for y in range(y1, y2):
            for x in range(x1, x2):
                rgb = image.getpixel((x, y))
                if self._is_skin(rgb):
                    pixels.append((x, y))
        return pixels

    def _is_skin(self, rgb: tuple[int, int, int]) -> bool:
        r, g, b = rgb
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return (
            r > 40
            and g > 20
            and b > 10
            and r > b
            and abs(r - g) > 4
            and 77 <= cb <= 135
            and 133 <= cr <= 180
            and y > 45
        )

    def _detect_landmarks(
        self,
        image: Image.Image,
        geometry: FaceGeometry,
    ) -> LandmarkDetection | None:
        if cv2 is None:
            return None
        facemark = self._get_facemark()
        if facemark is None:
            return None

        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rect = np.array(
            [[
                int(max(0, geometry.cx - geometry.fw * 0.70)),
                int(max(0, geometry.cy - geometry.fh * 0.82)),
                int(min(image.size[0] - 1, geometry.fw * 1.40)),
                int(min(image.size[1] - 1, geometry.fh * 1.72)),
            ]],
            dtype=np.int32,
        )
        ok, landmarks = facemark.fit(gray, rect)
        if not ok or landmarks is None or len(landmarks) == 0:
            return None
        points = np.array(landmarks[0][0], dtype=np.float32)
        x, y, w, h = rect[0].tolist()
        return LandmarkDetection(face_rect=(x, y, w, h), points=points)

    def _get_facemark(self):
        if self._facemark_load_attempted:
            return self._facemark
        self._facemark_load_attempted = True
        if cv2 is None or not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
            self._facemark = None
            return None
        model_path = "data/models/lbfmodel.yaml"
        try:
            facemark = cv2.face.createFacemarkLBF()
            facemark.loadModel(model_path)
            self._facemark = facemark
        except Exception:
            self._facemark = None
        return self._facemark

    def _build_region_masks_from_landmarks(
        self,
        image: Image.Image,
        geometry: FaceGeometry,
        detection: LandmarkDetection,
    ) -> dict[str, Image.Image]:
        size = image.size
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
        face_mask = self._polygon_mask(size, face_poly, blur_radius=2)
        left_brow_mask = self._polygon_mask(size, self._expand_poly(brow_left, 1.45, 1.8), blur_radius=1)
        right_brow_mask = self._polygon_mask(size, self._expand_poly(brow_right, 1.45, 1.8), blur_radius=1)
        eye_band_mask = self._polygon_mask(
            size,
            np.vstack([
                self._expand_poly(eye_left, 1.55, 2.0),
                self._expand_poly(eye_right, 1.55, 2.0),
            ]),
            blur_radius=2,
        )
        liner_left_mask = self._polygon_mask(size, self._expand_poly(eye_left, 1.28, 1.30), blur_radius=1)
        liner_right_mask = self._polygon_mask(size, self._expand_poly(eye_right, 1.28, 1.30), blur_radius=1)
        under_eye_mask = self._polygon_mask(
            size,
            self._under_eye_poly(eye_left, eye_right),
            blur_radius=2,
        )
        lips_mask = self._polygon_mask(size, mouth_outer, holes=[mouth_inner], blur_radius=1)
        blush_left_mask = self._polygon_mask(size, self._cheek_poly(jaw, nose, side="left"), blur_radius=4)
        blush_right_mask = self._polygon_mask(size, self._cheek_poly(jaw, nose, side="right"), blur_radius=4)
        contour_left_mask = self._polygon_mask(size, self._contour_poly(jaw, side="left"), blur_radius=4)
        contour_right_mask = self._polygon_mask(size, self._contour_poly(jaw, side="right"), blur_radius=4)
        nose_bridge_mask = self._polygon_mask(size, nose[0:4], blur_radius=1)
        forehead_highlight_mask = self._polygon_mask(size, self._forehead_poly(face_poly, geometry), blur_radius=4)
        nose_tip_mask = self._polygon_mask(size, nose[3:6], blur_radius=1)
        bangs_mask = self._polygon_mask(size, self._bangs_poly(face_poly, geometry), blur_radius=3)

        hair_mask = self._build_hair_mask_via_grabcut(image, face_poly, geometry)
        background_top = ImageChops.subtract(
            self._rect_mask(size, self._clamp_rect((0, 0, size[0], max(1, int(size[1] * 0.18))), size)),
            hair_mask,
        )
        background_side = ImageChops.subtract(
            self._rect_mask(
                size,
                self._clamp_rect((0, int(size[1] * 0.16), max(1, int(size[0] * 0.12)), int(size[1] * 0.90)), size),
            ),
            hair_mask,
        )

        return {
            "face": face_mask,
            "hair": hair_mask,
            "bangs": ImageChops.multiply(hair_mask, bangs_mask),
            "brow_left": left_brow_mask,
            "brow_right": right_brow_mask,
            "eye_band": eye_band_mask,
            "liner_left": liner_left_mask,
            "liner_right": liner_right_mask,
            "under_eye": under_eye_mask,
            "blush_left": blush_left_mask,
            "blush_right": blush_right_mask,
            "lips": lips_mask,
            "nose_bridge": nose_bridge_mask,
            "forehead_highlight": forehead_highlight_mask,
            "nose_tip": nose_tip_mask,
            "background_top": background_top,
            "background_side": background_side,
            "contour_left": contour_left_mask,
            "contour_right": contour_right_mask,
        }

    def _create_pixel_mask(
        self,
        image: Image.Image,
        rect: tuple[int, int, int, int],
        predicate,
    ) -> Image.Image:
        mask = Image.new("L", image.size, 0)
        pixels = mask.load()
        x1, y1, x2, y2 = rect
        for y in range(y1, y2):
            for x in range(x1, x2):
                rgb = image.getpixel((x, y))
                hsv = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
                if predicate(rgb, hsv, (x, y)):
                    pixels[x, y] = 255
        return mask

    def _polygon_mask(
        self,
        size: tuple[int, int],
        points: np.ndarray,
        holes: list[np.ndarray] | None = None,
        blur_radius: int = 0,
    ) -> Image.Image:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        outer = [tuple(map(float, p)) for p in points]
        if len(outer) >= 3:
            draw.polygon(outer, fill=255)
        for hole in holes or []:
            inner = [tuple(map(float, p)) for p in hole]
            if len(inner) >= 3:
                draw.polygon(inner, fill=0)
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask = mask.point(lambda value: 255 if value >= 40 else 0)
        return mask

    def _expand_poly(self, points: np.ndarray, x_scale: float, y_scale: float) -> np.ndarray:
        center = points.mean(axis=0)
        expanded = points.copy()
        expanded[:, 0] = center[0] + (expanded[:, 0] - center[0]) * x_scale
        expanded[:, 1] = center[1] + (expanded[:, 1] - center[1]) * y_scale
        return expanded

    def _under_eye_poly(self, eye_left: np.ndarray, eye_right: np.ndarray) -> np.ndarray:
        def build_eye_strip(eye: np.ndarray) -> np.ndarray:
            top = eye[[1, 2]]
            bottom = eye[[4, 5]]
            strip = np.vstack([
                top + np.array([0, 10]),
                bottom + np.array([0, 22]),
            ])
            return strip

        left = build_eye_strip(eye_left)
        right = build_eye_strip(eye_right)
        return np.vstack([left, right[::-1]])

    def _cheek_poly(self, jaw: np.ndarray, nose: np.ndarray, side: str) -> np.ndarray:
        if side == "left":
            return np.vstack([jaw[1], jaw[3], jaw[5], nose[2], nose[3], nose[4]])
        return np.vstack([jaw[15], jaw[13], jaw[11], nose[2], nose[3], nose[4]])

    def _contour_poly(self, jaw: np.ndarray, side: str) -> np.ndarray:
        if side == "left":
            return np.vstack([jaw[0], jaw[2], jaw[4], jaw[6], jaw[7]])
        return np.vstack([jaw[16], jaw[14], jaw[12], jaw[10], jaw[9]])

    def _forehead_poly(self, face_poly: np.ndarray, geometry: FaceGeometry) -> np.ndarray:
        x1 = float(face_poly[:, 0].min() + geometry.fw * 0.12)
        x2 = float(face_poly[:, 0].max() - geometry.fw * 0.12)
        y = float(face_poly[:, 1].min() - geometry.fh * 0.18)
        y2 = float(face_poly[:, 1].min() + geometry.fh * 0.05)
        cx = (x1 + x2) / 2.0
        return np.array(
            [
                [x1, y2],
                [cx - geometry.fw * 0.18, y],
                [cx + geometry.fw * 0.18, y],
                [x2, y2],
            ],
            dtype=np.float32,
        )

    def _bangs_poly(self, face_poly: np.ndarray, geometry: FaceGeometry) -> np.ndarray:
        x1 = float(face_poly[:, 0].min() + geometry.fw * 0.05)
        x2 = float(face_poly[:, 0].max() - geometry.fw * 0.05)
        y1 = float(face_poly[:, 1].min() - geometry.fh * 0.10)
        y2 = float(face_poly[:, 1].min() + geometry.fh * 0.20)
        return np.array([[x1, y2], [x1, y1], [x2, y1], [x2, y2]], dtype=np.float32)

    def _build_hair_mask_via_grabcut(
        self,
        image: Image.Image,
        face_poly: np.ndarray,
        geometry: FaceGeometry,
    ) -> Image.Image:
        size = image.size
        if cv2 is None:
            return self._build_region_masks_heuristic(image, geometry)["hair"]
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = np.full((size[1], size[0]), cv2.GC_BGD, dtype=np.uint8)
        head_rect = self._rect_from_geom(geometry, -1.20, -1.24, 1.20, 0.92)
        x1, y1, x2, y2 = head_rect
        mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

        face_mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, np.round(face_poly).astype(np.int32), 255)
        mask[face_mask > 0] = cv2.GC_FGD

        bg_strip = self._rect_from_geom(geometry, -1.50, 0.95, 1.50, 1.70)
        bx1, by1, bx2, by2 = bg_strip
        mask[by1:by2, bx1:bx2] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
            face_pil = Image.fromarray(face_mask, mode="L").filter(ImageFilter.MaxFilter(9))
            hair_pil = Image.fromarray(fg, mode="L")
            hair_only = ImageChops.subtract(hair_pil, face_pil)
            return self._merge_with_shape_hint(
                self._dilate_mask(hair_only, 2),
                self._build_region_masks_heuristic(image, geometry)["hair"],
                minimum_fill=0.08,
            )
        except Exception:
            return self._build_region_masks_heuristic(image, geometry)["hair"]

    def _merge_with_shape_hint(
        self,
        mask: Image.Image,
        hint: Image.Image,
        minimum_fill: float,
    ) -> Image.Image:
        if self._mask_fill_ratio(mask) >= minimum_fill:
            return self._dilate_mask(mask, 1)
        return hint

    def _mask_fill_ratio(self, mask: Image.Image) -> float:
        pixels = list(mask.getdata())
        if not pixels:
            return 0.0
        filled = sum(1 for p in pixels if p > 0)
        return filled / float(len(pixels))

    def _extract_masked_region(
        self,
        image: Image.Image,
        mask: Image.Image,
        fallback_rect: tuple[int, int, int, int],
    ) -> Image.Image:
        local_mask = mask.crop(fallback_rect)
        if local_mask.getbbox():
            masked = image.crop(fallback_rect).copy()
            masked.putalpha(local_mask)
            background = Image.new("RGBA", masked.size, (255, 255, 255, 255))
            composed = Image.alpha_composite(background, masked)
            return composed.convert("RGB")
        return image.crop(fallback_rect).convert("RGB")

    def _ellipse_mask(self, size: tuple[int, int], rect: tuple[int, int, int, int]) -> Image.Image:
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).ellipse(rect, fill=255)
        return mask

    def _rect_mask(self, size: tuple[int, int], rect: tuple[int, int, int, int]) -> Image.Image:
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle(rect, fill=255)
        return mask

    def _dilate_mask(self, mask: Image.Image, radius: int) -> Image.Image:
        if radius <= 0:
            return mask
        return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))

    def _erode_mask(self, mask: Image.Image, radius: int) -> Image.Image:
        if radius <= 0:
            return mask
        return mask.filter(ImageFilter.MinFilter(radius * 2 + 1))

    def _rect_from_geom(
        self,
        geometry: FaceGeometry,
        rx1: float,
        ry1: float,
        rx2: float,
        ry2: float,
    ) -> tuple[int, int, int, int]:
        rect = (
            int(geometry.cx + geometry.fw * rx1),
            int(geometry.cy + geometry.fh * ry1),
            int(geometry.cx + geometry.fw * rx2),
            int(geometry.cy + geometry.fh * ry2),
        )
        return self._clamp_rect(rect, (geometry.image_width, geometry.image_height))

    def _clamp_rect(
        self,
        rect: tuple[int, int, int, int],
        size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        width, height = size
        x1, y1, x2, y2 = rect
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        return (x1, y1, x2, y2)

    def _pick_hair_color_sample(
        self,
        image: Image.Image,
        regions: ParsedRegions,
        left_dark: float,
        right_dark: float,
    ) -> Image.Image:
        if right_dark >= left_dark:
            return self._blend_regions(regions.right_mid, regions.right_strip)
        return self._blend_regions(regions.left_strip, regions.right_mid)

    def _classify_hair_color(self, rgb: RGB) -> str:
        brightness = self._brightness(rgb)
        warmth = rgb.r - rgb.b
        if brightness < 0.20:
            return "black"
        if warmth > 18:
            return "dark_brown"
        if warmth > 8:
            return "natural_brown"
        return "black_brown"

    def _classify_hair_texture(self, image: Image.Image) -> str:
        edges = image.filter(ImageFilter.FIND_EDGES).convert("L")
        edge_std = ImageStat.Stat(edges).stddev[0]
        channel_std = self._channel_std(image)
        if edge_std > 38 or channel_std > 42:
            return "wavy"
        if edge_std > 26 or channel_std > 30:
            return "soft_wave"
        return "straight"

    def _classify_skin_finish(self, image: Image.Image) -> str:
        glow = self._highlight_ratio(image)
        if glow > 0.42:
            return "glowy"
        if glow > 0.24:
            return "semi_matte"
        return "matte"

    def _classify_makeup_color(self, rgb: RGB, role: str) -> str:
        hue = self._approx_hue(rgb)
        saturation = self._saturation(rgb)
        brightness = self._brightness(rgb)

        if role == "highlight":
            if brightness > 0.82:
                return "champagne"
            if brightness > 0.72:
                return "ivory"
            return "beige"

        if role == "contour":
            if brightness < 0.35:
                return "taupe_brown"
            return "soft_brown"

        if role == "brow":
            if brightness < 0.22:
                return "deep_charcoal"
            if hue < 24:
                return "deep_brown"
            return "natural_brown"

        if saturation < 0.14 and brightness > 0.72:
            return "soft_beige"
        if hue < 18:
            return "rose_brown" if role == "lips" else "warm_brown"
        if hue < 40:
            return "muted_peach" if role != "eyes" else "peach_brown"
        if hue < 70:
            return "soft_coral"
        if hue < 120:
            return "neutral_brown"
        if hue < 170:
            return "taupe"
        if hue < 260:
            return "cool_mauve"
        return "rose_pink"

    def _classify_secondary_eye_color(self, rgb: RGB) -> str:
        hue = self._approx_hue(rgb)
        if hue < 25:
            return "deep_brown"
        if hue < 55:
            return "soft_peach"
        if hue < 90:
            return "olive_taupe"
        if hue < 180:
            return "taupe"
        return "dusty_mauve"

    def _approx_hue(self, rgb: RGB) -> float:
        r = rgb.r / 255.0
        g = rgb.g / 255.0
        b = rgb.b / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        diff = mx - mn
        if diff == 0:
            return 0.0
        if mx == r:
            hue = (60 * ((g - b) / diff) + 360) % 360
        elif mx == g:
            hue = (60 * ((b - r) / diff) + 120) % 360
        else:
            hue = (60 * ((r - g) / diff) + 240) % 360
        return hue

    def _blend_regions(self, first: Image.Image, second: Image.Image) -> Image.Image:
        first_rgb = ImageOps.fit(first.convert("RGB"), (64, 64))
        second_rgb = ImageOps.fit(second.convert("RGB"), (64, 64))
        return Image.blend(first_rgb, second_rgb, alpha=0.5)

    def _dark_ratio(self, image: Image.Image) -> float:
        gray = image.convert("L")
        pixels = list(gray.getdata())
        if not pixels:
            return 0.0
        threshold = max(60, int(ImageStat.Stat(gray).mean[0] * 0.70))
        dark = sum(1 for p in pixels if p < threshold)
        return dark / float(len(pixels))

    def _median_rgb(self, image: Image.Image) -> RGB:
        resized = image.convert("RGB").resize((48, 48))
        pixels = sorted(resized.getdata(), key=lambda value: sum(value))
        mid = pixels[len(pixels) // 2]
        return RGB(*mid)

    def _median_rgb_non_background(self, image: Image.Image) -> RGB:
        resized = image.convert("RGB").resize((64, 64))
        filtered = [
            pixel
            for pixel in resized.getdata()
            if self._brightness(pixel) < 0.94 and max(pixel) - min(pixel) > 6
        ]
        if not filtered:
            return self._median_rgb(image)
        filtered.sort(key=lambda value: sum(value))
        mid = filtered[len(filtered) // 2]
        return RGB(*mid)

    def _rgb_to_hex(self, rgb: RGB) -> str:
        return "#{:02X}{:02X}{:02X}".format(int(rgb.r), int(rgb.g), int(rgb.b))

    def _channel_std(self, image: Image.Image) -> float:
        stat = ImageStat.Stat(image.convert("RGB"))
        return sum(stat.stddev) / len(stat.stddev)

    def _brightness(self, rgb: RGB | tuple[int, int, int]) -> float:
        if isinstance(rgb, tuple):
            r, g, b = rgb
        else:
            r, g, b = rgb.r, rgb.g, rgb.b
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    def _highlight_ratio(self, image: Image.Image) -> float:
        gray = image.convert("L")
        pixels = list(gray.getdata())
        if not pixels:
            return 0.0
        highlight = sum(1 for p in pixels if p > 210)
        return highlight / float(len(pixels))

    def _saturation(self, rgb: RGB) -> float:
        mx = max(rgb.r, rgb.g, rgb.b)
        mn = min(rgb.r, rgb.g, rgb.b)
        if mx == 0:
            return 0.0
        return round((mx - mn) / float(mx), 2)


reference_parser_service = ReferenceParserService()
