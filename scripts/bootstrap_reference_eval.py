import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "测试图"
EVAL_ROOT = ROOT / "evals" / "reference_parser"
ANNOTATIONS_DIR = EVAL_ROOT / "annotations"
MANIFEST_PATH = EVAL_ROOT / "manifest.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def iter_reference_images() -> list[Path]:
    return sorted(
        path for path in TEST_DIR.glob("reference*") if path.is_file()
    )


def build_annotation_template(image_path: Path) -> dict[str, object]:
    sample_id = image_path.stem
    return {
        "sample_id": sample_id,
        "image_path": str(image_path.resolve()),
        "status": "pending",
        "tags": [],
        "notes": "",
        "expected": {
            "hair": {
                "style": None,
                "updo_type": None,
                "primary_style": None,
                "parting": None,
                "surface_finish": None,
                "bun_silhouette": None,
                "color": None,
                "color_temperature": None,
            },
            "bangs": {
                "exists": None,
                "type": None,
            },
            "side_locks": {
                "exists": None,
            },
            "eyebrow": {
                "shape": None,
                "color": None,
                "tone": None,
            },
            "eyeshadow": {
                "upper_lid_color": None,
                "lower_lid_color": None,
                "outer_corner_color": None,
                "finish": None,
            },
            "eyeliner": {
                "style": None,
                "color": None,
            },
            "lips": {
                "color": None,
                "finish": None,
                "temperature": None,
            },
            "base_makeup": {
                "finish": None,
            },
        },
    }


def write_json_if_missing(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        return
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_status(annotation_path: Path) -> str:
    if not annotation_path.exists():
        return "missing"
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid"
    status = payload.get("status")
    return str(status) if status else "pending"


def main() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    reference_images = iter_reference_images()
    if not reference_images:
        raise RuntimeError("No reference images found in 测试图")

    manifest_samples: list[dict[str, object]] = []
    created_count = 0
    for image_path in reference_images:
        template = build_annotation_template(image_path)
        annotation_path = ANNOTATIONS_DIR / f"{image_path.stem}.json"
        if not annotation_path.exists():
            created_count += 1
        write_json_if_missing(annotation_path, template)
        manifest_samples.append(
            {
                "sample_id": image_path.stem,
                "image_path": str(image_path.resolve()),
                "annotation_path": str(annotation_path.resolve()),
                "status": load_status(annotation_path),
            }
        )

    manifest = {
        "generated_at": now_iso(),
        "source_dir": str(TEST_DIR.resolve()),
        "sample_count": len(manifest_samples),
        "created_annotation_count": created_count,
        "samples": manifest_samples,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
