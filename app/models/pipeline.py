from pydantic import BaseModel, Field

from app.models.job import Scores


class Pose(BaseModel):
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class MaskAsset(BaseModel):
    kind: str
    uri: str
    width: int = 1024
    height: int = 1024


class LandmarkPoint(BaseModel):
    x: float
    y: float


class FaceBBox(BaseModel):
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0


class FaceMeshAsset(BaseModel):
    uri: str = "mock://mesh/default.json"
    vertex_count: int = 468
    coordinate_system: str = "image_normalized"


class PreprocessResult(BaseModel):
    face_bbox: FaceBBox = Field(default_factory=FaceBBox)
    pose: Pose = Field(default_factory=Pose)
    landmarks_106: list[LandmarkPoint] = Field(default_factory=list)
    id_mask: MaskAsset = Field(
        default_factory=lambda: MaskAsset(kind="id_mask", uri="mock://mask/id.png")
    )
    style_mask: MaskAsset = Field(
        default_factory=lambda: MaskAsset(kind="style_mask", uri="mock://mask/style.png")
    )
    accessory_mask: MaskAsset = Field(
        default_factory=lambda: MaskAsset(kind="accessory_mask", uri="mock://mask/accessory.png")
    )
    editable_hair_mask: MaskAsset = Field(
        default_factory=lambda: MaskAsset(kind="editable_hair_mask", uri="mock://mask/editable_hair.png")
    )
    id_embedding: list[float] = Field(default_factory=list)
    face_mesh: FaceMeshAsset = Field(default_factory=FaceMeshAsset)
    accessory_tags: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class ColorFeature(BaseModel):
    label: str = "unknown"
    hex: str | None = None
    confidence: float = 0.5


class HairSideLocks(BaseModel):
    exists: bool = False
    length: str = "none"
    curl: float = 0.0
    intensity: float = 0.0


class HairFeatures(BaseModel):
    style: str = "unknown"
    updo_type: str = "unknown"
    length: str = "unknown"
    parting: str = "unknown"
    texture: str = "unknown"
    color: ColorFeature = Field(default_factory=ColorFeature)
    volume_crown: float = 0.5
    volume_side: float = 0.5
    hairline_exposure: float = 0.5
    side_locks: HairSideLocks = Field(default_factory=HairSideLocks)


class BangsFeatures(BaseModel):
    exists: bool = False
    type: str = "none"
    density: float = 0.0
    length: str = "none"
    curve: str = "none"
    gap_ratio: float = 0.0


class BaseMakeupFeatures(BaseModel):
    finish: str = "unknown"
    coverage: float = 0.5
    brightness_shift: float = 0.0
    evenness: float = 0.5
    glow: float = 0.0
    powderiness: float = 0.0
    intensity: float = 0.5


class BlushFeatures(BaseModel):
    color: str = "unknown"
    placement: str = "unknown"
    shape: str = "unknown"
    range: float = 0.0
    intensity: float = 0.0


class ContourFeatures(BaseModel):
    color: str = "unknown"
    nose_contour: float = 0.0
    cheek_contour: float = 0.0
    jaw_contour: float = 0.0
    intensity: float = 0.0


class HighlightFeatures(BaseModel):
    color: str = "unknown"
    nose_highlight: float = 0.0
    cheek_highlight: float = 0.0
    under_eye_highlight: float = 0.0
    intensity: float = 0.0


class EyebrowFeatures(BaseModel):
    shape: str = "unknown"
    color: str = "unknown"
    thickness: float = 0.0
    arch: float = 0.0
    hair_texture: float = 0.0
    intensity: float = 0.0


class EyelinerFeatures(BaseModel):
    color: str = "unknown"
    style: str = "unknown"
    length: float = 0.0
    thickness: float = 0.0
    tail_direction: str = "unknown"
    intensity: float = 0.0


class EyeshadowFeatures(BaseModel):
    main_color: str = "unknown"
    secondary_color: str = "unknown"
    placement: str = "unknown"
    gradient: str = "unknown"
    finish: str = "unknown"
    intensity: float = 0.0


class EyelashesFeatures(BaseModel):
    upper_density: float = 0.0
    lower_density: float = 0.0
    length: float = 0.0
    curl: float = 0.0
    cluster_style: str = "unknown"
    intensity: float = 0.0


class AegyoSalFeatures(BaseModel):
    exists: bool = False
    highlight_color: str = "unknown"
    shadow_color: str = "unknown"
    shape: str = "unknown"
    intensity: float = 0.0


class LipFeatures(BaseModel):
    color: str = "unknown"
    shape: str = "unknown"
    edge_blur: float = 0.0
    gloss: float = 0.0
    saturation: float = 0.0
    intensity: float = 0.0


class MakeupFeatures(BaseModel):
    base_makeup: BaseMakeupFeatures = Field(default_factory=BaseMakeupFeatures)
    blush: BlushFeatures = Field(default_factory=BlushFeatures)
    contour: ContourFeatures = Field(default_factory=ContourFeatures)
    highlight: HighlightFeatures = Field(default_factory=HighlightFeatures)
    eyebrow: EyebrowFeatures = Field(default_factory=EyebrowFeatures)
    eyeliner: EyelinerFeatures = Field(default_factory=EyelinerFeatures)
    eyeshadow: EyeshadowFeatures = Field(default_factory=EyeshadowFeatures)
    eyelashes: EyelashesFeatures = Field(default_factory=EyelashesFeatures)
    aegyo_sal: AegyoSalFeatures = Field(default_factory=AegyoSalFeatures)
    lips: LipFeatures = Field(default_factory=LipFeatures)


class TextureFeatures(BaseModel):
    skin_finish: str = "unknown"
    photo_style: str = "unknown"
    overall_vibe: str = "unknown"


class ReferenceParseResult(BaseModel):
    hair_features: HairFeatures = Field(default_factory=HairFeatures)
    bangs: BangsFeatures = Field(default_factory=BangsFeatures)
    makeup_features: MakeupFeatures = Field(default_factory=MakeupFeatures)
    texture_features: TextureFeatures = Field(default_factory=TextureFeatures)
    negative_constraints: list[str] = Field(default_factory=list)
    normalized_prompt_tokens: list[str] = Field(default_factory=list)
    parse_confidence: float = 0.5


class CandidateResult(BaseModel):
    candidate_id: str
    pipeline_type: str
    image_url: str
    metadata: dict[str, object] = Field(default_factory=dict)


class CandidateRecord(BaseModel):
    candidate_id: str
    job_id: str
    pipeline_type: str
    image_url: str
    is_selected: bool = False
    scores: Scores = Field(default_factory=Scores)
    metadata: dict[str, object] = Field(default_factory=dict)
