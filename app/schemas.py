"""Input models for product-description generation."""

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
