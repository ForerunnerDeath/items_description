"""FastAPI entrypoint for safe product-description generation."""

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.generator import generate_product_description
from app.llm_client import LLMClientError, OpenAICompatibleLLMClient
from app.preprocessor import prepare_product
from app.schemas import GenerationResponse, ProductInput
from app.validator import build_generation_response

app = FastAPI(title="Items Description", version="0.1.0")


def build_llm_client() -> OpenAICompatibleLLMClient:
    """Create a client only for a product that needs LLM generation."""
    settings = get_settings()
    if not settings.model.strip() or (
        not settings.api_key.strip() and not settings.base_url.strip()
    ):
        raise HTTPException(
            status_code=503,
            detail="LLM backend is not configured.",
        )

    return OpenAICompatibleLLMClient(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_tokens=settings.max_tokens,
        timeout_seconds=settings.timeout_seconds,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerationResponse)
async def generate(product: ProductInput) -> GenerationResponse:
    """Return a draft with warnings, or a safe HTTP error on backend failure."""
    prepared = prepare_product(product)
    if prepared.blocked or not prepared.can_generate:
        return build_generation_response(prepared)

    client = build_llm_client()
    try:
        return await generate_product_description(product, client)
    except LLMClientError as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM generation failed.",
        ) from exc
    finally:
        await client.close()
