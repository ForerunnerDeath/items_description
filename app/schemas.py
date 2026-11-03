"""Input and output models for product-description generation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProductInput(BaseModel):
    """Product facts supplied by the caller; no LLM output fields yet."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    name: str = Field(min_length=1)
    manufacturer: str | None = None
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""


class GenerationResponse(BaseModel):
    """Public result, including incomplete and review-required generations.

    Content limits and factual checks belong to the later validation stage.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    status: Literal["generated", "needs_review", "insufficient_data"]
    seo_title: str | None = None
    seo_description: str | None = None
    characteristics: dict[str, str] = Field(default_factory=dict)
    search_queries: list[str] = Field(default_factory=list)
    commercial_block: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
