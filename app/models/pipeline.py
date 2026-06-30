from typing import Literal

from pydantic import BaseModel, Field

from app.models.job import Scores


class MainlineCapabilityProfile(BaseModel):
    mainline_mode: Literal["native_executable", "hybrid", "text_reference_only", "disabled"] = (
        "text_reference_only"
    )
    supports_executable_masks: bool = False
    supports_control_image: bool = False
    supports_identity_embedding: bool = False
    supports_reference_image: bool = True
    supports_multi_image_reference: bool = True
    evidence_level: Literal["official_confirmed", "workspace_assumed", "unknown"] = "unknown"
    confirmed_surfaces: list[str] = Field(default_factory=list)
    missing_surfaces: list[str] = Field(default_factory=list)
    control_surface: Literal["native_controls", "hybrid_controls", "text_reference_only", "disabled"] = (
        "text_reference_only"
    )
    summary: str = ""


class PipelineDecision(BaseModel):
    primary_pipeline: str
    fallback_pipeline: str | None = None
    reason: str = ""
    capability_mode: str = "unknown"
    capability_profile: MainlineCapabilityProfile | None = None


class PipelineAttempt(BaseModel):
    pipeline: str
    status: str
    reason: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


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
    source_landmark_count: int = 0
    coverage_ratio: float = 0.0


class RegionImageAsset(BaseModel):
    kind: str
    uri: str
    width: int = 1024
    height: int = 1024


class ReferenceRegionAssets(BaseModel):
    hair_patch: RegionImageAsset | None = None
    bangs_patch: RegionImageAsset | None = None
    eyes_patch: RegionImageAsset | None = None
    upper_lid_patch: RegionImageAsset | None = None
    lower_lid_patch: RegionImageAsset | None = None
    outer_corner_patch: RegionImageAsset | None = None
    brows_patch: RegionImageAsset | None = None
    lips_patch: RegionImageAsset | None = None
    cheeks_patch: RegionImageAsset | None = None
    complexion_patch: RegionImageAsset | None = None


class IdentityEmbeddingAsset(BaseModel):
    vector: list[float] = Field(default_factory=list)
    provider: str = "pseudo_preview"
    dimension: int = 0
    source: str = "face_lock_mask"
    confidence: float = 0.0


class GenerationStrengthControls(BaseModel):
    makeup_strength: float = 0.75
    hairstyle_strength: float = 0.85
    identity_lock_strength: float = 0.95
    preserve_accessories: bool = True


class RegionBlendProfile(BaseModel):
    source_weight: float = 0.0
    style_weight: float = 1.0
    notes: str = ""


class RegionGatingPolicy(BaseModel):
    strategy: str = "static_defaults"
    face_core: RegionBlendProfile = Field(default_factory=RegionBlendProfile)
    feature_lock: RegionBlendProfile = Field(default_factory=RegionBlendProfile)
    contour: RegionBlendProfile = Field(default_factory=RegionBlendProfile)
    accessory: RegionBlendProfile = Field(default_factory=RegionBlendProfile)
    hair: RegionBlendProfile = Field(default_factory=RegionBlendProfile)
    makeup: RegionBlendProfile = Field(default_factory=RegionBlendProfile)
    stage_overrides: dict[str, GenerationStrengthControls] = Field(default_factory=dict)
    reasoning: list[str] = Field(default_factory=list)


class QualityGate(BaseModel):
    identity_threshold: float = 0.92
    accessory_threshold: float = 0.8
    transfer_threshold: float = 0.7
    artifact_penalty_threshold: float = 0.2
    max_retry_count: int = 3
    reject_on_identity_failure: bool = True
    reject_on_accessory_failure: bool = True


class PreprocessResult(BaseModel):
    source_image_ref: str = ""
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
        default_factory=lambda: MaskAsset(
            kind="editable_hair_mask",
            uri="mock://mask/editable_hair.png",
        )
    )
    editable_makeup_mask: MaskAsset = Field(
        default_factory=lambda: MaskAsset(
            kind="editable_makeup_mask",
            uri="mock://mask/editable_makeup.png",
        )
    )
    face_lock_mask: MaskAsset = Field(
        default_factory=lambda: MaskAsset(
            kind="face_lock_mask",
            uri="mock://mask/face_lock.png",
        )
    )
    feature_lock_mask: MaskAsset = Field(
        default_factory=lambda: MaskAsset(
            kind="feature_lock_mask",
            uri="mock://mask/feature_lock.png",
        )
    )
    contour_lock_mask: MaskAsset = Field(
        default_factory=lambda: MaskAsset(
            kind="contour_lock_mask",
            uri="mock://mask/contour_lock.png",
        )
    )
    id_embedding: IdentityEmbeddingAsset = Field(default_factory=IdentityEmbeddingAsset)
    face_mesh: FaceMeshAsset = Field(default_factory=FaceMeshAsset)
    accessory_tags: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class GenerationControlBundle(BaseModel):
    source_image: str
    reference_image: str
    mode: str = "full_transfer"
    pipeline_variant: str = "two_stage_local_edit"
    delivery_mode: Literal["native_controls", "hybrid_controls", "text_reference_only", "fallback_only"] = (
        "text_reference_only"
    )
    id_mask: MaskAsset
    style_mask: MaskAsset
    accessory_mask: MaskAsset
    face_lock_mask: MaskAsset
    feature_lock_mask: MaskAsset
    contour_lock_mask: MaskAsset
    editable_hair_mask: MaskAsset
    editable_makeup_mask: MaskAsset
    face_bbox: FaceBBox = Field(default_factory=FaceBBox)
    pose: Pose = Field(default_factory=Pose)
    landmarks_106: list[LandmarkPoint] = Field(default_factory=list)
    face_mesh: FaceMeshAsset = Field(default_factory=FaceMeshAsset)
    identity_embedding: IdentityEmbeddingAsset = Field(default_factory=IdentityEmbeddingAsset)
    controls: GenerationStrengthControls = Field(default_factory=GenerationStrengthControls)
    region_gating_policy: RegionGatingPolicy | None = None
    quality_gate: QualityGate = Field(default_factory=QualityGate)
    capability_profile: MainlineCapabilityProfile | None = None


class RegionMaskSet(BaseModel):
    hair: str | None = None
    bangs: str | None = None
    hairline: str | None = None
    brow_left: str | None = None
    brow_right: str | None = None
    upper_eyelid_left: str | None = None
    upper_eyelid_right: str | None = None
    lower_eyelid_left: str | None = None
    lower_eyelid_right: str | None = None
    outer_corner_left: str | None = None
    outer_corner_right: str | None = None
    aegyo_sal_left: str | None = None
    aegyo_sal_right: str | None = None
    eyelashes_upper: str | None = None
    eyelashes_lower: str | None = None
    lips: str | None = None
    blush_left: str | None = None
    blush_right: str | None = None
    nose_highlight: str | None = None
    contour_left: str | None = None
    contour_right: str | None = None


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
    color_temperature: str = "unknown"
    color_depth: str = "unknown"
    volume_crown: float = 0.5
    volume_side: float = 0.5
    hairline_exposure: float = 0.5
    side_locks: HairSideLocks = Field(default_factory=HairSideLocks)
    primary_style: str = "unknown"
    secondary_style: str = "unknown"
    finish: str = "unknown"
    surface_finish: str = "unknown"
    bun_silhouette: str = "none"
    gloss: float = 0.0
    sleekness: float = 0.0
    confidence: float = 0.5


class BangsFeatures(BaseModel):
    exists: bool = False
    type: str = "none"
    density: float = 0.0
    length: str = "none"
    curve: str = "none"
    gap_ratio: float = 0.0
    confidence: float = 0.5


class BaseMakeupFeatures(BaseModel):
    finish: str = "unknown"
    coverage: float = 0.5
    brightness_shift: float = 0.0
    evenness: float = 0.5
    glow: float = 0.0
    powderiness: float = 0.0
    intensity: float = 0.5
    concealer_coverage: float = 0.0
    brightness_level: float = 0.0
    finish_confidence: float = 0.5


class BlushFeatures(BaseModel):
    color: str = "unknown"
    placement: str = "unknown"
    shape: str = "unknown"
    range: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.5


class ContourFeatures(BaseModel):
    color: str = "unknown"
    nose_contour: float = 0.0
    cheek_contour: float = 0.0
    jaw_contour: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.5


class HighlightFeatures(BaseModel):
    color: str = "unknown"
    nose_highlight: float = 0.0
    cheek_highlight: float = 0.0
    under_eye_highlight: float = 0.0
    forehead_highlight: float = 0.0
    nose_tip_highlight: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.5


class EyebrowFeatures(BaseModel):
    shape: str = "unknown"
    color: str = "unknown"
    tone: str = "unknown"
    thickness: float = 0.0
    density: float = 0.0
    arch: float = 0.0
    hair_texture: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.5


class EyelinerFeatures(BaseModel):
    color: str = "unknown"
    style: str = "unknown"
    length: float = 0.0
    thickness: float = 0.0
    tail_direction: str = "unknown"
    intensity: float = 0.0
    confidence: float = 0.5


class EyeshadowFeatures(BaseModel):
    main_color: str = "unknown"
    secondary_color: str = "unknown"
    upper_lid_color: str = "unknown"
    lower_lid_color: str = "unknown"
    outer_corner_color: str = "unknown"
    placement: str = "unknown"
    gradient: str = "unknown"
    finish: str = "unknown"
    shimmer: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.5


class EyelashesFeatures(BaseModel):
    upper_density: float = 0.0
    lower_density: float = 0.0
    length: float = 0.0
    curl: float = 0.0
    cluster_style: str = "unknown"
    intensity: float = 0.0
    confidence: float = 0.5


class AegyoSalFeatures(BaseModel):
    exists: bool = False
    highlight_color: str = "unknown"
    shadow_color: str = "unknown"
    shape: str = "unknown"
    intensity: float = 0.0
    confidence: float = 0.5


class LipFeatures(BaseModel):
    color: str = "unknown"
    shape: str = "unknown"
    temperature: str = "unknown"
    lightness: str = "unknown"
    finish: str = "unknown"
    edge_blur: float = 0.0
    gloss: float = 0.0
    saturation: float = 0.0
    intensity: float = 0.0
    edge_definition: float = 0.0
    cupid_bow_definition: float = 0.0
    bite_effect: float = 0.0
    confidence: float = 0.5


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
    confidence: float = 0.5


class ReferenceParseResult(BaseModel):
    region_masks: RegionMaskSet = Field(default_factory=RegionMaskSet)
    region_assets: ReferenceRegionAssets = Field(default_factory=ReferenceRegionAssets)
    hair_features: HairFeatures = Field(default_factory=HairFeatures)
    bangs: BangsFeatures = Field(default_factory=BangsFeatures)
    makeup_features: MakeupFeatures = Field(default_factory=MakeupFeatures)
    texture_features: TextureFeatures = Field(default_factory=TextureFeatures)
    style_caption: str = ""
    consistency_flags: list[str] = Field(default_factory=list)
    field_confidence_overrides: dict[str, float] = Field(default_factory=dict)
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
