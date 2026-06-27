import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


def is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def is_data_url(value: str) -> bool:
    return value.startswith("data:image/")


def is_local_file(value: str) -> bool:
    return Path(value).exists() and Path(value).is_file()


def normalize_image_input(value: str) -> tuple[str, str]:
    """
    Return a tuple of:
    - transport kind: url | base64
    - transport value
    """
    if is_http_url(value):
        return "url", value

    if is_data_url(value):
        _, payload = value.split(",", 1)
        return "base64", payload

    if is_local_file(value):
        raw = Path(value).read_bytes()
        return "base64", base64.b64encode(raw).decode("utf-8")

    return "base64", value


def load_image_bytes(value: str) -> bytes:
    if is_local_file(value):
        return Path(value).read_bytes()
    if is_data_url(value):
        _, payload = value.split(",", 1)
        return base64.b64decode(payload)
    return base64.b64decode(value)


def normalize_image_pair_to_base64(
    first: str,
    second: str,
    size: tuple[int, int] = (1024, 1024),
    background: tuple[int, int, int] = (255, 255, 255),
) -> tuple[str, str]:
    first_image = Image.open(io.BytesIO(load_image_bytes(first))).convert("RGB")
    second_image = Image.open(io.BytesIO(load_image_bytes(second))).convert("RGB")
    first_norm = _fit_image(first_image, size, background)
    second_norm = _fit_image(second_image, size, background)
    return _to_base64(first_norm), _to_base64(second_norm)


def _fit_image(
    image: Image.Image,
    size: tuple[int, int],
    background: tuple[int, int, int],
) -> Image.Image:
    contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    x = (size[0] - contained.width) // 2
    y = (size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def _to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
