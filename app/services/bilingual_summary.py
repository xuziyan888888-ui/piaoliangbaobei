from __future__ import annotations

from typing import Any

from app.models.job import JobRecord
from app.models.pipeline import ReferenceParseResult


def _entry(
    label_en: str,
    label_zh: str,
    value_en: Any,
    value_zh: Any | None = None,
) -> dict[str, Any]:
    return {
        "label_en": label_en,
        "label_zh": label_zh,
        "value_en": value_en,
        "value_zh": value_en if value_zh is None else value_zh,
    }


class BilingualSummaryService:
    def build_reference_extraction_summary(
        self,
        reference: ReferenceParseResult,
    ) -> dict[str, Any]:
        hair = reference.hair_features
        bangs = reference.bangs
        makeup = reference.makeup_features
        texture = reference.texture_features
        return {
            "title_en": "Reference extraction result",
            "title_zh": "参考图提取结果",
            "sections": [
                {
                    "section_en": "Hair",
                    "section_zh": "发型",
                    "items": [
                        _entry("Style", "发型大类", hair.style, self._zh_hair_style(hair.style)),
                        _entry(
                            "Primary style",
                            "主发型结构",
                            hair.primary_style,
                            self._zh_hair_style(hair.primary_style),
                        ),
                        _entry(
                            "Secondary style",
                            "次发型结构",
                            hair.secondary_style,
                            self._zh_updo_type(hair.secondary_style),
                        ),
                        _entry("Updo type", "盘发类型", hair.updo_type, self._zh_updo_type(hair.updo_type)),
                        _entry("Length", "长度", hair.length, self._zh_length(hair.length)),
                        _entry("Parting", "分缝", hair.parting, self._zh_parting(hair.parting)),
                        _entry("Texture", "发丝质感", hair.texture, self._zh_texture(hair.texture)),
                        _entry("Hair color", "发色", hair.color.label, self._zh_color(hair.color.label)),
                        _entry("Hair color temperature", "发色冷暖", hair.color_temperature, self._zh_temperature(hair.color_temperature)),
                        _entry("Hair color depth", "发色深浅", hair.color_depth, self._zh_depth(hair.color_depth)),
                        _entry("Hair surface finish", "发面质感", hair.surface_finish, self._zh_hair_finish(hair.surface_finish)),
                        _entry("Bun silhouette", "盘发轮廓", hair.bun_silhouette, self._zh_bun_silhouette(hair.bun_silhouette)),
                        _entry("Crown volume", "颅顶蓬松度", hair.volume_crown),
                        _entry("Side volume", "两侧蓬松度", hair.volume_side),
                        _entry("Hairline exposure", "发际线露出程度", hair.hairline_exposure),
                        _entry("Gloss", "发丝光泽", hair.gloss),
                        _entry("Sleekness", "服帖度", hair.sleekness),
                    ],
                },
                {
                    "section_en": "Bangs",
                    "section_zh": "刘海",
                    "items": [
                        _entry("Has bangs", "是否有刘海", bangs.exists, "有" if bangs.exists else "无"),
                        _entry("Bang type", "刘海类型", bangs.type, self._zh_bangs_type(bangs.type)),
                        _entry("Density", "刘海密度", bangs.density),
                        _entry("Length", "刘海长度", bangs.length, self._zh_bang_length(bangs.length)),
                        _entry("Curve", "刘海弯度", bangs.curve, self._zh_curve(bangs.curve)),
                        _entry("Gap ratio", "留白比例", bangs.gap_ratio),
                    ],
                },
                {
                    "section_en": "Makeup",
                    "section_zh": "妆容",
                    "items": [
                        _entry(
                            "Base finish",
                            "底妆质感",
                            makeup.base_makeup.finish,
                            self._zh_base_finish(makeup.base_makeup.finish),
                        ),
                        _entry("Base intensity", "底妆强度", makeup.base_makeup.intensity),
                        _entry("Base brightness shift", "底妆提亮偏移", makeup.base_makeup.brightness_shift),
                        _entry("Base glow", "底妆光泽", makeup.base_makeup.glow),
                        _entry("Base powderiness", "底妆粉感", makeup.base_makeup.powderiness),
                        _entry("Blush color", "腮红颜色", makeup.blush.color, self._zh_color(makeup.blush.color)),
                        _entry("Blush intensity", "腮红强度", makeup.blush.intensity),
                        _entry("Contour color", "修容颜色", makeup.contour.color, self._zh_color(makeup.contour.color)),
                        _entry("Highlight color", "高光颜色", makeup.highlight.color, self._zh_color(makeup.highlight.color)),
                        _entry("Eyebrow shape", "眉型", makeup.eyebrow.shape, self._zh_brow_shape(makeup.eyebrow.shape)),
                        _entry("Eyebrow color", "眉色", makeup.eyebrow.color, self._zh_color(makeup.eyebrow.color)),
                        _entry("Eyebrow tone", "眉色冷暖", makeup.eyebrow.tone, self._zh_temperature(makeup.eyebrow.tone)),
                        _entry("Upper lid color", "上眼皮颜色", makeup.eyeshadow.upper_lid_color, self._zh_color(makeup.eyeshadow.upper_lid_color)),
                        _entry("Lower lid color", "下眼皮颜色", makeup.eyeshadow.lower_lid_color, self._zh_color(makeup.eyeshadow.lower_lid_color)),
                        _entry("Outer corner color", "眼尾加深色", makeup.eyeshadow.outer_corner_color, self._zh_color(makeup.eyeshadow.outer_corner_color)),
                        _entry("Eyeshadow color", "眼影主色", makeup.eyeshadow.main_color, self._zh_color(makeup.eyeshadow.main_color)),
                        _entry(
                            "Eyeshadow secondary",
                            "眼影辅色",
                            makeup.eyeshadow.secondary_color,
                            self._zh_color(makeup.eyeshadow.secondary_color),
                        ),
                        _entry("Eyeshadow finish", "眼影材质", makeup.eyeshadow.finish, self._zh_makeup_finish(makeup.eyeshadow.finish)),
                        _entry("Eyeliner style", "眼线样式", makeup.eyeliner.style, self._zh_eyeliner_style(makeup.eyeliner.style)),
                        _entry("Eyeliner color", "眼线颜色", makeup.eyeliner.color, self._zh_color(makeup.eyeliner.color)),
                        _entry("Lash intensity", "睫毛强度", makeup.eyelashes.intensity),
                        _entry("Aegyo sal", "卧蚕", makeup.aegyo_sal.exists, "有" if makeup.aegyo_sal.exists else "无"),
                        _entry("Lip color", "唇色", makeup.lips.color, self._zh_color(makeup.lips.color)),
                        _entry("Lip finish", "唇妆质地", makeup.lips.finish, self._zh_makeup_finish(makeup.lips.finish)),
                        _entry("Lip temperature", "唇色冷暖", makeup.lips.temperature, self._zh_temperature(makeup.lips.temperature)),
                        _entry("Lip lightness", "唇色深浅", makeup.lips.lightness, self._zh_depth(makeup.lips.lightness)),
                        _entry("Lip gloss", "唇部光泽", makeup.lips.gloss),
                        _entry("Lip saturation", "唇部饱和度", makeup.lips.saturation),
                        _entry("Lip edge definition", "唇缘清晰度", makeup.lips.edge_definition),
                    ],
                },
                {
                    "section_en": "Photo texture",
                    "section_zh": "照片质感",
                    "items": [
                        _entry("Photo style", "照片风格", texture.photo_style, self._zh_photo_style(texture.photo_style)),
                        _entry("Overall vibe", "整体氛围", texture.overall_vibe, self._zh_overall_vibe(texture.overall_vibe)),
                        _entry("Style caption", "整体语义描述", reference.style_caption),
                        _entry("Parse confidence", "解析置信度", reference.parse_confidence),
                    ],
                },
            ],
        }

    def build_transfer_payload_summary(
        self,
        job: JobRecord,
        reference: ReferenceParseResult,
    ) -> dict[str, Any]:
        hair = reference.hair_features
        bangs = reference.bangs
        makeup = reference.makeup_features
        texture = reference.texture_features
        return {
            "title_en": "Transferred style payload sent to generation",
            "title_zh": "发送给生成模型的妆发负载",
            "sections": [
                {
                    "section_en": "Control params",
                    "section_zh": "控制参数",
                    "items": [
                        _entry("Mode", "模式", job.mode, self._zh_mode(job.mode)),
                        _entry("Makeup strength", "妆容迁移强度", job.makeup_strength),
                        _entry("Hairstyle strength", "发型迁移强度", job.hairstyle_strength),
                        _entry("Identity lock strength", "身份锁定强度", job.identity_lock_strength),
                        _entry(
                            "Preserve accessories",
                            "是否保留配饰",
                            job.preserve_accessories,
                            "保留" if job.preserve_accessories else "不保留",
                        ),
                    ],
                },
                {
                    "section_en": "Hair payload",
                    "section_zh": "发型负载",
                    "items": [
                        _entry("Target hair style", "目标发型", hair.style, self._zh_hair_style(hair.style)),
                        _entry(
                            "Target primary style",
                            "目标主发型结构",
                            hair.primary_style,
                            self._zh_hair_style(hair.primary_style),
                        ),
                        _entry(
                            "Target secondary style",
                            "目标次发型结构",
                            hair.secondary_style,
                            self._zh_updo_type(hair.secondary_style),
                        ),
                        _entry("Target updo type", "目标盘发类型", hair.updo_type, self._zh_updo_type(hair.updo_type)),
                        _entry("Target length", "目标长度", hair.length, self._zh_length(hair.length)),
                        _entry("Target parting", "目标分缝", hair.parting, self._zh_parting(hair.parting)),
                        _entry("Target texture", "目标发丝质感", hair.texture, self._zh_texture(hair.texture)),
                        _entry("Target hair surface finish", "目标发面质感", hair.surface_finish, self._zh_hair_finish(hair.surface_finish)),
                        _entry("Target bun silhouette", "目标盘发轮廓", hair.bun_silhouette, self._zh_bun_silhouette(hair.bun_silhouette)),
                        _entry("Target hair color", "目标发色", hair.color.label, self._zh_color(hair.color.label)),
                        _entry("Target hair temperature", "目标发色冷暖", hair.color_temperature, self._zh_temperature(hair.color_temperature)),
                        _entry(
                            "Target bangs",
                            "目标刘海",
                            bangs.type if bangs.exists else "none",
                            self._zh_bangs_type(bangs.type) if bangs.exists else "无刘海",
                        ),
                    ],
                },
                {
                    "section_en": "Makeup payload",
                    "section_zh": "妆容负载",
                    "items": [
                        _entry("Base makeup finish", "底妆质感", makeup.base_makeup.finish, self._zh_base_finish(makeup.base_makeup.finish)),
                        _entry("Base makeup intensity", "底妆强度", makeup.base_makeup.intensity),
                        _entry("Base makeup glow", "底妆光泽", makeup.base_makeup.glow),
                        _entry("Blush target", "腮红目标", makeup.blush.color, self._zh_color(makeup.blush.color)),
                        _entry("Contour target", "修容目标", makeup.contour.color, self._zh_color(makeup.contour.color)),
                        _entry("Highlight target", "高光目标", makeup.highlight.color, self._zh_color(makeup.highlight.color)),
                        _entry(
                            "Eyebrow target",
                            "眉毛目标",
                            f"{makeup.eyebrow.shape} / {makeup.eyebrow.color} / {makeup.eyebrow.tone}",
                            f"{self._zh_brow_shape(makeup.eyebrow.shape)} / {self._zh_color(makeup.eyebrow.color)} / {self._zh_temperature(makeup.eyebrow.tone)}",
                        ),
                        _entry("Upper lid target", "上眼皮目标", makeup.eyeshadow.upper_lid_color, self._zh_color(makeup.eyeshadow.upper_lid_color)),
                        _entry("Lower lid target", "下眼皮目标", makeup.eyeshadow.lower_lid_color, self._zh_color(makeup.eyeshadow.lower_lid_color)),
                        _entry("Outer corner target", "眼尾目标", makeup.eyeshadow.outer_corner_color, self._zh_color(makeup.eyeshadow.outer_corner_color)),
                        _entry("Eyeshadow finish", "眼影材质目标", makeup.eyeshadow.finish, self._zh_makeup_finish(makeup.eyeshadow.finish)),
                        _entry(
                            "Eyeliner target",
                            "眼线目标",
                            f"{makeup.eyeliner.style} / {makeup.eyeliner.color}",
                            f"{self._zh_eyeliner_style(makeup.eyeliner.style)} / {self._zh_color(makeup.eyeliner.color)}",
                        ),
                        _entry("Lash target", "睫毛目标", makeup.eyelashes.intensity),
                        _entry("Aegyo sal target", "卧蚕目标", makeup.aegyo_sal.intensity),
                        _entry("Lip target", "唇妆目标", makeup.lips.color, self._zh_color(makeup.lips.color)),
                        _entry("Lip finish target", "唇妆质地目标", makeup.lips.finish, self._zh_makeup_finish(makeup.lips.finish)),
                        _entry("Lip gloss target", "唇部光泽目标", makeup.lips.gloss),
                        _entry("Lip edge target", "唇缘清晰度目标", makeup.lips.edge_definition),
                    ],
                },
                {
                    "section_en": "Texture payload",
                    "section_zh": "质感负载",
                    "items": [
                        _entry("Photo style target", "照片风格目标", texture.photo_style, self._zh_photo_style(texture.photo_style)),
                        _entry("Overall vibe target", "整体氛围目标", texture.overall_vibe, self._zh_overall_vibe(texture.overall_vibe)),
                        _entry("Style caption", "整体语义描述", reference.style_caption),
                    ],
                },
            ],
        }

    def _zh_hair_style(self, value: str) -> str:
        return {
            "down": "披发",
            "updo": "盘发",
            "half_up": "半扎发",
            "slicked_back_updo": "后梳贴头盘发",
            "tight_bun": "紧致发髻",
            "unknown": "未知",
        }.get(value, value)

    def _zh_updo_type(self, value: str) -> str:
        return {
            "none": "无",
            "bun_or_ponytail": "丸子头或马尾类盘发",
            "half_up_clip": "半扎夹发",
            "tight_bun": "紧致发髻",
            "unknown": "未知",
        }.get(value, value)

    def _zh_length(self, value: str) -> str:
        return {
            "short": "短发",
            "medium": "中长发",
            "long": "长发",
            "unknown": "未知",
        }.get(value, value)

    def _zh_parting(self, value: str) -> str:
        return {
            "middle": "中分",
            "side_3_7": "三七分",
            "side_4_6": "四六分",
            "side_6_4": "六四分",
            "side_7_3": "七三分",
            "none_or_natural_back": "无明显分缝或自然后梳",
            "unclear": "分缝不明确",
            "unknown": "未知",
        }.get(value, value)

    def _zh_texture(self, value: str) -> str:
        return {
            "straight": "直发",
            "soft_wave": "轻微波浪",
            "wavy": "明显波浪",
            "straight_sleek": "顺直服帖",
            "unknown": "未知",
        }.get(value, value)

    def _zh_hair_finish(self, value: str) -> str:
        return {
            "sleek_polished": "顺滑抛光感",
            "soft_lustrous": "柔和光泽感",
            "soft_natural": "柔和自然感",
            "airy_textured": "轻盈纹理感",
            "balanced_natural": "均衡自然感",
            "unknown": "未知",
        }.get(value, value)

    def _zh_bun_silhouette(self, value: str) -> str:
        return {
            "none": "无",
            "tight_crown_bun": "紧致颅顶发髻",
            "soft_crown_bun": "柔和颅顶发髻",
            "lifted_updo": "上提式盘发",
            "rounded_updo": "圆润盘发",
            "unknown": "未知",
        }.get(value, value)

    def _zh_bangs_type(self, value: str) -> str:
        return {
            "none": "无刘海",
            "airy_side_bangs": "空气感侧分刘海",
            "side_swept_bangs": "侧分斜刘海",
            "see_through_bangs": "空气刘海",
            "soft_bangs": "柔和刘海",
        }.get(value, value)

    def _zh_bang_length(self, value: str) -> str:
        return {
            "none": "无",
            "brow_to_eye": "眉眼之间",
        }.get(value, value)

    def _zh_curve(self, value: str) -> str:
        return {
            "none": "无",
            "soft": "柔和弯度",
        }.get(value, value)

    def _zh_base_finish(self, value: str) -> str:
        return {
            "matte": "雾面",
            "semi_matte": "半雾面",
            "glowy": "光泽奶油肌",
            "semi_glowy": "半光泽奶油肌",
            "unknown": "未知",
        }.get(value, value)

    def _zh_makeup_finish(self, value: str) -> str:
        return {
            "matte": "哑光",
            "satin": "缎光",
            "shimmer": "珠光",
            "glossy": "镜面光泽",
            "pearly": "珠贝光泽",
            "unknown": "未知",
        }.get(value, value)

    def _zh_temperature(self, value: str) -> str:
        return {
            "warm": "暖调",
            "cool": "冷调",
            "neutral": "中性调",
            "unknown": "未知",
        }.get(value, value)

    def _zh_depth(self, value: str) -> str:
        return {
            "deep": "深色",
            "medium_deep": "中深色",
            "medium": "中色",
            "light": "浅色",
            "very_light": "很浅",
            "unknown": "未知",
        }.get(value, value)

    def _zh_brow_shape(self, value: str) -> str:
        return {
            "soft_arch": "柔和弯眉",
            "straight_soft": "柔和平眉",
            "straight_soft_arch": "平直微弯眉",
            "unknown": "未知",
        }.get(value, value)

    def _zh_eyeliner_style(self, value: str) -> str:
        return {
            "micro_wing": "微上挑眼线",
            "inner_defined": "内眼线强调",
            "thin_lifted": "细长上扬眼线",
            "unknown": "未知",
        }.get(value, value)

    def _zh_photo_style(self, value: str) -> str:
        return {
            "clean_studio_portrait": "干净棚拍人像",
            "soft_portrait": "柔和人像",
            "natural_photo": "自然照片感",
            "unknown": "未知",
        }.get(value, value)

    def _zh_overall_vibe(self, value: str) -> str:
        return {
            "clean_commute": "干净通勤感",
            "soft_lifestyle": "柔和生活感",
            "daily_photo": "日常照片感",
            "unknown": "未知",
        }.get(value, value)

    def _zh_mode(self, value: str) -> str:
        return {
            "full_transfer": "完整迁移",
            "hair_only": "仅迁移发型",
            "makeup_only": "仅迁移妆容",
        }.get(value, value)

    def _zh_color(self, value: str) -> str:
        return {
            "dark_brown": "深棕",
            "warm_brown": "暖棕",
            "soft_brown": "柔和棕",
            "deep_brown": "深棕褐",
            "black_brown": "黑棕",
            "natural_brown": "自然棕",
            "ivory": "象牙白",
            "peach_brown": "蜜桃棕",
            "rose_pink": "玫瑰粉",
            "black": "黑色",
            "muted_pink": "低饱和粉",
            "cool_pink": "冷粉",
            "peach": "蜜桃色",
            "coral": "珊瑚色",
            "red_brown": "红棕",
            "grey_brown": "灰棕",
            "deep_charcoal": "深炭灰",
            "taupe_brown": "灰棕修容",
            "cool_taupe": "冷灰褐",
            "rose_brown": "玫瑰棕",
            "muted_peach": "低饱和蜜桃",
            "soft_beige": "柔雾米色",
            "soft_beige_brown": "米棕",
            "soft_coral": "柔和珊瑚",
            "neutral_brown": "中性棕",
            "taupe": "灰褐",
            "cool_mauve": "冷灰紫粉",
            "champagne": "香槟色",
            "beige": "米色",
            "pearly_apricot": "珠光杏粉",
            "dusty_rose": "烟粉玫瑰",
            "beige_brown": "米棕",
            "soft_peach": "柔和蜜桃",
            "olive_taupe": "橄榄灰褐",
            "unknown": "未知",
        }.get(value, value)


bilingual_summary_service = BilingualSummaryService()
