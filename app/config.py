import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(filename: str = ".env.local") -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = root / filename
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


@dataclass
class ArkHTTPConfig:
    base_url: str = ""
    model: str = ""
    generation_route: str = "global"
    auth_mode: str = "aksk"
    access_key: str = ""
    secret_key: str = ""
    action: str = "CVSync2AsyncSubmitTask"
    get_action: str = "CVSync2AsyncGetResult"
    version: str = "2022-08-31"
    region: str = "cn-north-1"
    service: str = "cv"
    timeout_seconds: float = 120.0
    poll_interval_seconds: float = 3.0
    max_poll_attempts: int = 25
    inpaint_action: str = "CVSync2AsyncSubmitTask"
    inpaint_get_action: str = "CVSync2AsyncGetResult"
    inpaint_model: str = ""
    mask_transport_mode: str = "binary_append"
    disable_logo: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.access_key and self.secret_key)


@dataclass
class LocalInpaintConfig:
    provider: str = "mock"
    endpoint: str = ""
    api_key: str = ""
    auth_header: str = "Authorization"
    timeout_seconds: float = 45.0
    model_name: str = "jimeng_image_4_0"

    @property
    def enabled(self) -> bool:
        return self.provider != "mock" and bool(self.endpoint)


@dataclass
class AppConfig:
    local_inpaint: LocalInpaintConfig
    ark_http: ArkHTTPConfig


def load_config() -> AppConfig:
    generation_provider = os.getenv("GENERATION_PROVIDER", "").strip().lower()
    makeup_provider = os.getenv("AI_MAKEUP_PROVIDER", "").strip().lower()
    resolved_provider = "mock"
    if generation_provider:
        resolved_provider = generation_provider
    if makeup_provider:
        resolved_provider = makeup_provider

    return AppConfig(
        local_inpaint=LocalInpaintConfig(
            provider=os.getenv("PBB_LOCAL_INPAINT_PROVIDER", resolved_provider or "mock"),
            endpoint=os.getenv("PBB_LOCAL_INPAINT_URL", "").strip(),
            api_key=os.getenv("PBB_LOCAL_INPAINT_API_KEY", "").strip(),
            auth_header=os.getenv("PBB_LOCAL_INPAINT_AUTH_HEADER", "Authorization").strip(),
            timeout_seconds=float(os.getenv("PBB_LOCAL_INPAINT_TIMEOUT_SECONDS", "45")),
            model_name=os.getenv(
                "PBB_LOCAL_INPAINT_MODEL",
                os.getenv("ARK_INPAINT_MODEL", "jimeng_image_4_0"),
            ).strip(),
        ),
        ark_http=ArkHTTPConfig(
            base_url=os.getenv("ARK_IMAGE_EDIT_URL", "").strip(),
            model=os.getenv("ARK_MODEL", "").strip(),
            generation_route=os.getenv("ARK_GENERATION_ROUTE", "global").strip(),
            auth_mode=os.getenv("ARK_AUTH_MODE", "aksk").strip(),
            access_key=os.getenv("ARK_ACCESS_KEY", "").strip(),
            secret_key=os.getenv("ARK_SECRET_KEY", "").strip(),
            action=os.getenv("ARK_ACTION", "CVSync2AsyncSubmitTask").strip(),
            get_action=os.getenv("ARK_GET_ACTION", "CVSync2AsyncGetResult").strip(),
            version=os.getenv("ARK_VERSION", "2022-08-31").strip(),
            region=os.getenv("ARK_REGION", "cn-north-1").strip(),
            service=os.getenv("ARK_SERVICE", "cv").strip(),
            timeout_seconds=float(os.getenv("ARK_TIMEOUT_SECONDS", "120")),
            poll_interval_seconds=float(os.getenv("ARK_POLL_INTERVAL_SECONDS", "3")),
            max_poll_attempts=int(os.getenv("ARK_MAX_POLL_ATTEMPTS", "25")),
            inpaint_action=os.getenv("ARK_INPAINT_ACTION", "CVSync2AsyncSubmitTask").strip(),
            inpaint_get_action=os.getenv("ARK_INPAINT_GET_ACTION", "CVSync2AsyncGetResult").strip(),
            inpaint_model=os.getenv("ARK_INPAINT_MODEL", "").strip(),
            mask_transport_mode=os.getenv("ARK_MASK_TRANSPORT_MODE", "binary_append").strip(),
            disable_logo=os.getenv("ARK_DISABLE_LOGO", "true").strip().lower() in {"1", "true", "yes"},
        ),
    )


settings = load_config()
