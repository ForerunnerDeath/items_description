"""Deterministic SEO checks and safe assembly of the public response."""

from app.grounding import validate_grounding
from app.preprocessor import PreparedProduct
from app.schemas import GenerationResponse, LLMGeneratedContent


def validate_seo_content(content: LLMGeneratedContent) -> list[str]:
    """Check mechanical SEO constraints, not factual truth of generated prose."""
    warnings: list[str] = []

    title_length = len(content.seo_title)
    if not 1 <= title_length <= 70:
        warnings.append("SEO-заголовок должен содержать от 1 до 70 символов.")

    description_length = len(content.seo_description)
    if not 800 <= description_length <= 1200:
        warnings.append("SEO-описание должно содержать от 800 до 1200 символов.")

    query_count = len(content.search_queries)
    if not 3 <= query_count <= 5:
        warnings.append("Необходимо от 3 до 5 поисковых запросов.")

    normalized_queries = [
        " ".join(query.split()).casefold() for query in content.search_queries
    ]
    if any(not query for query in normalized_queries):
        warnings.append("Поисковые запросы не должны быть пустыми.")
    if len(set(normalized_queries)) != len(normalized_queries):
        warnings.append("Поисковые запросы должны быть уникальными.")

    if not content.commercial_block.strip():
        warnings.append("Коммерческий блок не должен быть пустым.")

    return warnings


def validate_generated_content(
    prepared: PreparedProduct, content: LLMGeneratedContent
) -> list[str]:
    """Combine mechanical SEO checks with lexical grounding warnings."""
    return [*validate_seo_content(content), *validate_grounding(prepared, content)]


def build_generation_response(
    prepared: PreparedProduct,
    generated: LLMGeneratedContent | None = None,
) -> GenerationResponse:
    """Copy trusted specifications and annotate SEO and grounding concerns.

    The status "generated" means these checks passed, not that every statement
    has been independently verified or semantically fact-checked.
    """
    if not prepared.can_generate:
        if generated is not None:
            raise ValueError("LLM content must not exist for insufficient product data")
        return GenerationResponse(
            status="insufficient_data",
            characteristics=prepared.product.specs.copy(),
            missing_fields=list(prepared.missing_fields),
            warnings=list(prepared.warnings),
        )

    if generated is None:
        raise ValueError("LLM content is required when product facts are available")

    warnings = [*prepared.warnings, *validate_generated_content(prepared, generated)]
    return GenerationResponse(
        status="needs_review" if warnings else "generated",
        seo_title=generated.seo_title,
        seo_description=generated.seo_description,
        characteristics=prepared.product.specs.copy(),
        search_queries=generated.search_queries.copy(),
        commercial_block=generated.commercial_block,
        missing_fields=list(prepared.missing_fields),
        warnings=warnings,
    )
