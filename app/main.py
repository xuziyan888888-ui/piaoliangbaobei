from fastapi import FastAPI

from app.api.routes import router as api_router

app = FastAPI(
    title="AI Makeup Transfer API",
    version="0.1.0",
    description="MVP backend skeleton for identity-preserving makeup and hairstyle transfer.",
)

app.include_router(api_router)


@app.get("/healthz", tags=["system"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
