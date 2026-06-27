from app.models.pipeline import FaceBBox, LandmarkPoint, MaskAsset, Pose, PreprocessResult


class PreprocessService:
    def run(self, source_image: str) -> PreprocessResult:
        return PreprocessResult(
            face_bbox=FaceBBox(x1=128, y1=96, x2=896, y2=960),
            pose=Pose(yaw=0.02, pitch=-0.03, roll=0.01),
            landmarks_106=[
                LandmarkPoint(x=0.32, y=0.38),
                LandmarkPoint(x=0.68, y=0.38),
                LandmarkPoint(x=0.50, y=0.54),
                LandmarkPoint(x=0.39, y=0.72),
                LandmarkPoint(x=0.61, y=0.72),
            ],
            id_mask=MaskAsset(kind="id_mask", uri=f"mock://preprocess/{hash(source_image)}/id_mask.png"),
            style_mask=MaskAsset(kind="style_mask", uri=f"mock://preprocess/{hash(source_image)}/style_mask.png"),
            accessory_mask=MaskAsset(
                kind="accessory_mask", uri=f"mock://preprocess/{hash(source_image)}/accessory_mask.png"
            ),
            editable_hair_mask=MaskAsset(
                kind="editable_hair_mask",
                uri=f"mock://preprocess/{hash(source_image)}/editable_hair_mask.png",
            ),
            id_embedding=[0.01 * ((idx % 10) - 5) for idx in range(32)],
            accessory_tags=["glasses"],
            quality_flags=["frontal_face", "strong_identity_lock_recommended"],
        )


preprocess_service = PreprocessService()
