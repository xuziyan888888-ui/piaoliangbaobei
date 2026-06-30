import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.reference_parser import reference_parser_service


EVAL_ROOT = ROOT / "evals" / "reference_parser"
MANIFEST_PATH = EVAL_ROOT / "manifest.json"

FIELD_SPECS = [
    ("hair.style", ("hair_features", "style")),
    ("hair.updo_type", ("hair_features", "updo_type")),
    ("hair.primary_style", ("hair_features", "primary_style")),
    ("hair.parting", ("hair_features", "parting")),
    ("hair.surface_finish", ("hair_features", "surface_finish")),
    ("hair.bun_silhouette", ("hair_features", "bun_silhouette")),
    ("hair.color", ("hair_features", "color", "label")),
    ("hair.color_temperature", ("hair_features", "color_temperature")),
    ("bangs.exists", ("bangs", "exists")),
    ("bangs.type", ("bangs", "type")),
    ("side_locks.exists", ("hair_features", "side_locks", "exists")),
    ("eyebrow.shape", ("makeup_features", "eyebrow", "shape")),
    ("eyebrow.color", ("makeup_features", "eyebrow", "color")),
    ("eyebrow.tone", ("makeup_features", "eyebrow", "tone")),
    ("eyeshadow.upper_lid_color", ("makeup_features", "eyeshadow", "upper_lid_color")),
    ("eyeshadow.lower_lid_color", ("makeup_features", "eyeshadow", "lower_lid_color")),
    ("eyeshadow.outer_corner_color", ("makeup_features", "eyeshadow", "outer_corner_color")),
    ("eyeshadow.finish", ("makeup_features", "eyeshadow", "finish")),
    ("eyeliner.style", ("makeup_features", "eyeliner", "style")),
    ("eyeliner.color", ("makeup_features", "eyeliner", "color")),
    ("lips.color", ("makeup_features", "lips", "color")),
    ("lips.finish", ("makeup_features", "lips", "finish")),
    ("lips.temperature", ("makeup_features", "lips", "temperature")),
    ("base_makeup.finish", ("makeup_features", "base_makeup", "finish")),
]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(payload: dict[str, object], dotted_path: str):
    current = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def get_actual(payload: dict[str, object], path_parts: tuple[str, ...]):
    current = payload
    for part in path_parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def is_known(value) -> bool:
    return value not in {None, "", "unknown", "unclear"}


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Manifest not found. Run scripts/bootstrap_reference_eval.py first.")

    manifest = load_json(MANIFEST_PATH)
    samples = manifest.get("samples", [])

    summary_counter = Counter()
    coverage_counter = Counter()
    field_match_counter = Counter()
    per_sample: list[dict[str, object]] = []

    for sample in samples:
        image_path = Path(str(sample["image_path"]))
        annotation_path = Path(str(sample["annotation_path"]))
        annotation = load_json(annotation_path)
        expected = annotation.get("expected", {})
        result = reference_parser_service.run(str(image_path))
        actual = result.model_dump(mode="json")

        sample_matches = 0
        sample_total = 0
        mismatches: list[dict[str, object]] = []
        matched_fields: list[str] = []

        for expected_path, actual_path in FIELD_SPECS:
            expected_value = get_nested(expected, expected_path)
            if not is_known(expected_value):
                continue
            coverage_counter[expected_path] += 1
            sample_total += 1
            actual_value = get_actual(actual, actual_path)
            if actual_value == expected_value:
                sample_matches += 1
                field_match_counter[expected_path] += 1
                matched_fields.append(expected_path)
            else:
                mismatches.append(
                    {
                        "field": expected_path,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )

        status = str(annotation.get("status") or "pending")
        if sample_total > 0:
            summary_counter["scored_samples"] += 1
            summary_counter["annotated_fields"] += sample_total
            summary_counter["matched_fields"] += sample_matches
        if status == "ready":
            summary_counter["ready_samples"] += 1
        elif status == "approved":
            summary_counter["approved_samples"] += 1
        else:
            summary_counter["pending_samples"] += 1

        per_sample.append(
            {
                "sample_id": sample["sample_id"],
                "status": status,
                "annotated_field_count": sample_total,
                "matched_field_count": sample_matches,
                "accuracy": round(sample_matches / sample_total, 4) if sample_total else None,
                "matched_fields": matched_fields,
                "mismatches": mismatches,
            }
        )

    coverage = {
        field: {
            "annotated_count": coverage_counter[field],
            "matched_count": field_match_counter[field],
            "accuracy": (
                round(field_match_counter[field] / coverage_counter[field], 4)
                if coverage_counter[field]
                else None
            ),
        }
        for field, _ in FIELD_SPECS
    }

    total_fields = summary_counter["annotated_fields"]
    matched_fields = summary_counter["matched_fields"]
    report = {
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "sample_count": len(samples),
        "scored_samples": summary_counter["scored_samples"],
        "ready_samples": summary_counter["ready_samples"],
        "approved_samples": summary_counter["approved_samples"],
        "pending_samples": summary_counter["pending_samples"],
        "annotated_field_count": total_fields,
        "matched_field_count": matched_fields,
        "overall_accuracy": round(matched_fields / total_fields, 4) if total_fields else None,
        "coverage_by_field": coverage,
        "samples": per_sample,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
