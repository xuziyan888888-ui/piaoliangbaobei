import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.job import JobCreateRequest
from app.services.orchestrator import orchestrator
from app.storage.memory_store import store


TEST_DIR = Path(r"D:\水木年华\测试图")


def pick_random_images() -> tuple[Path, Path]:
    source_candidates = sorted(TEST_DIR.glob("origin*"))
    reference_candidates = sorted(TEST_DIR.glob("reference*"))
    if not source_candidates:
        raise RuntimeError("No source images found in 测试图")
    if not reference_candidates:
        raise RuntimeError("No reference images found in 测试图")
    return random.choice(source_candidates), random.choice(reference_candidates)


def main() -> None:
    source_image, reference_image = pick_random_images()
    payload = JobCreateRequest(
        source_image=str(source_image),
        reference_image=str(reference_image),
        mode="full_transfer",
        preserve_accessories=True,
        candidate_count=1,
    )
    job = orchestrator.create_job(payload)
    result = orchestrator.run_job(job.job_id)
    artifacts = {
        "job": result.model_dump(mode="json"),
        "candidates": [
            candidate.model_dump(mode="json") for candidate in store.get_candidates(job.job_id)
        ],
        "preprocess_result": store.get_preprocess_result(job.job_id).model_dump(mode="json")
        if store.get_preprocess_result(job.job_id)
        else None,
        "reference_parse_result": store.get_reference_parse_result(job.job_id).model_dump(mode="json")
        if store.get_reference_parse_result(job.job_id)
        else None,
        "reference_extraction_summary": result.metadata.get("reference_extraction_summary"),
        "transfer_payload_summary": result.metadata.get("transfer_payload_summary"),
        "provider_prompt": result.metadata.get("provider_prompt"),
        "source_image": str(source_image),
        "reference_image": str(reference_image),
    }
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
