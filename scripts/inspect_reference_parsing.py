import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.reference_parser import reference_parser_service


def iter_reference_images(args: list[str]) -> list[Path]:
    if args:
        return [Path(arg) for arg in args]
    test_dir = ROOT / "测试图"
    return sorted(test_dir.glob("reference*"))


def main() -> None:
    paths = iter_reference_images(sys.argv[1:])
    if not paths:
        raise RuntimeError("No reference images found.")

    payload = {}
    for path in paths:
        result = reference_parser_service.run(str(path))
        payload[str(path)] = result.model_dump(mode="json")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
