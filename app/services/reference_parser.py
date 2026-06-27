from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageOps, ImageStat

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
    ReferenceParseResult,
    TextureFeatures,
)
from app.utils.images import load_image_bytes


@dataclass
class RGB:
    r: float
    g: float
    b: float


class ReferenceParserService:
    def run(self, reference_image: str) -> ReferenceParseResult:
        pil_image = Image.open(io.BytesIO(load_image_bytes(reference_image))).convert("RGB")

        hair = self._parse_hair(pil_image)
        bangs = self._parse_bangs(pil_image, hair)
        makeup = self._parse_makeup(pil_image)
        texture = self._parse_texture(pil_image)
        confidence = self._aggregate_confidence(hair, bangs, makeup, texture)

        return ReferenceParseResult(
            hair_features=hair,
            bangs=bangs,
            makeup_features=makeup,
            texture_features=texture,
            negative_constraints=[
                "do not inherit reference identity",
                "do not inherit reference clothing",
                "do not inherit reference background",
            ],
            normalized_prompt_tokens=self._build_tokens(hair, bangs, makeup, texture),
            parse_confidence=confidence,
        )

    def _parse_hair(self, image: Image.Image) -> HairFeatures:
        left_strip = self._crop(image, 0.03, 0.14, 0.24, 0.94)
        right_strip = self._crop(image, 0.76, 0.14, 0.97, 0.94)
        upper_band = self._crop(image, 0.20, 0.00, 0.82, 0.18)
        lower_band = self._crop(image, 0.14, 0.62, 0.86, 0.98)
        crown_band = self._crop(image, 0.28, 0.00, 0.72, 0.12)
        side_band = self._crop(image, 0.08, 0.18, 0.92, 0.48)
        left_top = self._crop(image, 0.16, 0.06, 0.42, 0.32)
        right_top = self._crop(image, 0.58, 0.06, 0.84, 0.32)
        right_mid = self._crop(image, 0.60, 0.30, 0.90, 0.84)

        left_dark = self._dark_ratio(left_strip)
        right_dark = self._dark_ratio(right_strip)
        upper_dark = self._dark_ratio(upper_band)
        lower_dark = self._dark_ratio(lower_band)
        crown_dark = self._dark_ratio(crown_band)
        right_mid_dark = self._dark_ratio(right_mid)

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

        left_top_dark = self._dark_ratio(left_top)
        right_top_dark = self._dark_ratio(right_top)
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

        texture = self._classify_hair_texture(right_mid)
        volume_crown = min(0.95, max(0.10, crown_dark * 1.45))
        volume_side = min(0.95, max(0.10, self._dark_ratio(side_band) * 1.05))
        hairline_exposure = round(max(0.10, min(0.90, 1.0 - self._dark_ratio(self._crop(image, 0.36, 0.10, 0.64, 0.24)))), 2)

        hair_sample = self._pick_hair_color_sample(image, left_dark, right_dark)
        hair_rgb = self._median_rgb(hair_sample)
        hair_color = ColorFeature(
            label=self._classify_hair_color(hair_rgb),
            hex=self._rgb_to_hex(hair_rgb),
            confidence=0.78,
        )

        side_lock_exists = style == "down" and (left_dark > 0.12 or right_dark > 0.12)
        side_lock_length = "medium" if side_lock_exists and length != "short" else "none"
        side_lock_intensity = 0.52 if side_lock_exists else 0.0

        if style == "updo":
            hairline_exposure = max(hairline_exposure, 0.58)

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
        )

    def _parse_bangs(self, image: Image.Image, hair: HairFeatures) -> BangsFeatures:
        forehead_band = self._crop(image, 0.28, 0.09, 0.72, 0.27)
        center_forehead = self._crop(image, 0.43, 0.09, 0.57, 0.27)
        left_forehead = self._crop(image, 0.29, 0.10, 0.44, 0.28)
        right_forehead = self._crop(image, 0.56, 0.10, 0.71, 0.28)

        forehead_dark = self._dark_ratio(forehead_band)
        center_dark = self._dark_ratio(center_forehead)
        left_dark = self._dark_ratio(left_forehead)
        right_dark = self._dark_ratio(right_forehead)
        exists = forehead_dark > 0.12 or center_dark > 0.08 or abs(left_dark - right_dark) > 0.10

        if not exists:
            return BangsFeatures(
                exists=False,
                type="none",
                density=0.0,
                length="none",
                curve="none",
                gap_ratio=0.0,
            )

        asymmetry = abs(left_dark - right_dark)
        gap_ratio = round(max(0.0, min(1.0, 1.0 - center_dark / max(forehead_dark, 0.001))), 2)

        if hair.parting.startswith("side") and (gap_ratio > 0.08 or asymmetry > 0.05):
            bangs_type = "airy_side_bangs"
            density = 0.34
        elif asymmetry > 0.10:
            bangs_type = "side_swept_bangs"
            density = 0.44
        elif gap_ratio > 0.35:
            bangs_type = "see_through_bangs"
            density = 0.32
        else:
            bangs_type = "soft_bangs"
            density = 0.50

        return BangsFeatures(
            exists=True,
            type=bangs_type,
            density=round(density, 2),
            length="brow_to_eye",
            curve="soft",
            gap_ratio=gap_ratio,
        )

    def _parse_makeup(self, image: Image.Image) -> MakeupFeatures:
        face_center = self._crop(image, 0.30, 0.21, 0.70, 0.58)
        left_cheek = self._crop(image, 0.18, 0.42, 0.35, 0.60)
        right_cheek = self._crop(image, 0.65, 0.42, 0.82, 0.60)
        lips = self._crop(image, 0.37, 0.61, 0.63, 0.76)
        left_brow = self._crop(image, 0.24, 0.25, 0.42, 0.34)
        right_brow = self._crop(image, 0.58, 0.25, 0.76, 0.34)
        eye_band = self._crop(image, 0.22, 0.28, 0.78, 0.46)
        liner_left = self._crop(image, 0.23, 0.31, 0.43, 0.39)
        liner_right = self._crop(image, 0.57, 0.31, 0.77, 0.39)
        under_eye = self._crop(image, 0.26, 0.40, 0.74, 0.50)
        nose_bridge = self._crop(image, 0.46, 0.32, 0.54, 0.58)

        face_rgb = self._median_rgb(face_center)
        cheek_rgb = self._median_rgb(self._blend_regions(left_cheek, right_cheek))
        lip_rgb = self._median_rgb(lips)
        brow_rgb = self._median_rgb(self._blend_regions(left_brow, right_brow))
        eye_rgb = self._median_rgb(eye_band)
        under_eye_rgb = self._median_rgb(under_eye)
        nose_rgb = self._median_rgb(nose_bridge)

        face_brightness = self._brightness(face_rgb)
        face_glow = self._highlight_ratio(face_center)
        face_evenness = min(0.96, max(0.20, 1.0 - self._channel_std(face_center) / 95.0))
        lip_saturation = self._saturation(lip_rgb)
        blush_saturation = self._saturation(cheek_rgb)
        eye_saturation = self._saturation(eye_rgb)
        brow_darkness = 1.0 - self._brightness(brow_rgb)
        liner_density = max(self._dark_ratio(liner_left), self._dark_ratio(liner_right))
        under_eye_std = self._channel_std(under_eye)

        base_finish = self._classify_skin_finish(face_center)
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
            ),
            blush=BlushFeatures(
                color=blush_color,
                placement="mid_cheek",
                shape="soft_oval",
                range=0.34,
                intensity=round(max(0.18, min(0.88, blush_saturation + 0.12)), 2),
            ),
            contour=ContourFeatures(
                color=contour_color,
                nose_contour=round(max(0.08, min(0.78, (1.0 - self._brightness(nose_rgb)) * 0.8)), 2),
                cheek_contour=round(max(0.06, min(0.74, (1.0 - self._brightness(cheek_rgb)) * 0.55)), 2),
                jaw_contour=0.18,
                intensity=round(max(0.10, min(0.68, (1.0 - self._brightness(nose_rgb)) * 0.65)), 2),
            ),
            highlight=HighlightFeatures(
                color=highlight_color,
                nose_highlight=round(max(0.10, min(0.78, face_glow + 0.12)), 2),
                cheek_highlight=round(max(0.10, min(0.76, face_glow + 0.08)), 2),
                under_eye_highlight=round(max(0.10, min(0.72, 0.26 + under_eye_std / 80.0)), 2),
                intensity=round(max(0.10, min(0.80, face_glow + 0.10)), 2),
            ),
            eyebrow=EyebrowFeatures(
                shape="soft_arch" if brow_darkness > 0.26 else "straight_soft",
                color=brow_color,
                thickness=round(max(0.16, min(0.84, brow_darkness)), 2),
                arch=round(max(0.10, min(0.72, 0.30 + self._channel_std(self._blend_regions(left_brow, right_brow)) / 80.0)), 2),
                hair_texture=round(max(0.10, min(0.76, self._channel_std(self._blend_regions(left_brow, right_brow)) / 60.0)), 2),
                intensity=round(max(0.18, min(0.90, brow_darkness + 0.14)), 2),
            ),
            eyeliner=EyelinerFeatures(
                color="black" if liner_density > 0.20 else "soft_brown",
                style="micro_wing" if liner_density > 0.28 else "inner_defined",
                length=round(max(0.12, min(0.88, 0.24 + liner_density * 1.4)), 2),
                thickness=round(max(0.08, min(0.72, 0.12 + liner_density * 0.9)), 2),
                tail_direction="slightly_up",
                intensity=round(max(0.10, min(0.92, 0.18 + liner_density * 1.6)), 2),
            ),
            eyeshadow=EyeshadowFeatures(
                main_color=eye_color,
                secondary_color=self._classify_secondary_eye_color(eye_rgb),
                placement="upper_lid_outer_corner",
                gradient="soft_gradient",
                finish="satin" if eye_saturation > 0.18 else "matte",
                intensity=round(max(0.12, min(0.82, eye_saturation + 0.10)), 2),
            ),
            eyelashes=EyelashesFeatures(
                upper_density=round(max(0.16, min(0.92, liner_density * 2.1 + 0.10)), 2),
                lower_density=round(max(0.05, min(0.55, liner_density * 0.8)), 2),
                length=round(max(0.16, min(0.88, liner_density * 1.8 + 0.14)), 2),
                curl=round(max(0.12, min(0.84, 0.22 + self._channel_std(eye_band) / 65.0)), 2),
                cluster_style="natural_separated",
                intensity=round(max(0.14, min(0.88, liner_density * 1.6 + 0.12)), 2),
            ),
            aegyo_sal=AegyoSalFeatures(
                exists=under_eye_std > 10,
                highlight_color=highlight_color,
                shadow_color=contour_color,
                shape="soft_parallel",
                intensity=round(max(0.06, min(0.68, under_eye_std / 28.0)), 2),
            ),
            lips=LipFeatures(
                color=lip_color,
                shape="soft_full",
                edge_blur=0.46,
                gloss=round(max(0.06, min(0.88, face_glow * 0.70 + 0.08)), 2),
                saturation=round(lip_saturation, 2),
                intensity=round(max(0.22, min(0.92, lip_saturation + 0.16)), 2),
            ),
        )

    def _parse_texture(self, image: Image.Image) -> TextureFeatures:
        top_bg = self._crop(image, 0.00, 0.00, 1.00, 0.14)
        side_bg = self._crop(image, 0.00, 0.18, 0.10, 0.86)
        bg_mix = self._blend_regions(top_bg, side_bg)
        bg_std = self._channel_std(bg_mix)
        bg_brightness = self._brightness(self._median_rgb(bg_mix))
        face_center = self._crop(image, 0.30, 0.21, 0.70, 0.58)

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
            skin_finish=self._classify_skin_finish(face_center),
            photo_style=photo_style,
            overall_vibe=overall_vibe,
        )

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

    def _pick_hair_color_sample(self, image: Image.Image, left_dark: float, right_dark: float) -> Image.Image:
        if right_dark >= left_dark:
            return self._crop(image, 0.62, 0.24, 0.92, 0.84)
        return self._crop(image, 0.08, 0.24, 0.38, 0.84)

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

    def _crop(self, image: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
        width, height = image.size
        left = int(max(0, min(width, width * x1)))
        top = int(max(0, min(height, height * y1)))
        right = int(max(left + 1, min(width, width * x2)))
        bottom = int(max(top + 1, min(height, height * y2)))
        return image.crop((left, top, right, bottom))

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

    def _rgb_to_hex(self, rgb: RGB) -> str:
        return "#{:02X}{:02X}{:02X}".format(int(rgb.r), int(rgb.g), int(rgb.b))

    def _channel_std(self, image: Image.Image) -> float:
        stat = ImageStat.Stat(image.convert("RGB"))
        return sum(stat.stddev) / len(stat.stddev)

    def _brightness(self, rgb: RGB) -> float:
        return (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255.0

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
