"""Coordinate bounded LLM retries for deterministic SEO validation failures."""

from app.llm_client import LLMClient, LLMClientError
from app.preprocessor import prepare_product
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.schemas import GenerationResponse, ProductInput
from app.validator import build_generation_response, validate_generated_content

MAX_RETRIES = 2

_RETRYABLE_RESPONSE_ERRORS = {
    "InvalidResponse": "Ответ не соответствует обязательной JSON-схеме.",
    "EmptyResponse": "Ответ не содержит структурированного JSON.",
    "IncompleteResponse": "Ответ был обрезан и не содержит всех обязательных полей.",
}


def _retry_prompt(original_prompt: str, issues: list[str]) -> str:
    """Return only trusted validation feedback, never the previous LLM output."""
    feedback = "\n".join(f"- {issue}" for issue in issues)
    return (
        f"{original_prompt}\n\n"
        "Предыдущий результат не прошёл проверку требований.\n"
        "Создай новый полный JSON-ответ и исправь все перечисленные ошибки.\n"
        "Сохраняй факты из исходного JSON, ничего не придумывай. "
        "Если данных недостаточно для соблюдения ограничений, "
        "не добавляй неподтверждённых утверждений.\n"
        f"Ошибки проверки:\n{feedback}"
    )


async def generate_product_description(
    product: ProductInput, client: LLMClient
) -> GenerationResponse:
    """Generate content, retry twice on SEO defects, or return needs_review.

    Missing source information is not fixable by regeneration. Malformed
    model responses get bounded retries; provider/refusal errors propagate.
    """
    prepared = prepare_product(product)
    if not prepared.can_generate:
        return build_generation_response(prepared)

    original_prompt = build_user_prompt(prepared)
    user_prompt = original_prompt
    retries = 0

    while True:
        try:
            generated = await client.generate_structured(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except LLMClientError as exc:
            feedback = _RETRYABLE_RESPONSE_ERRORS.get(exc.error_type)
            if feedback is None or retries >= MAX_RETRIES:
                raise
            user_prompt = _retry_prompt(original_prompt, [feedback])
            retries += 1
            continue

        response = build_generation_response(prepared, generated)
        issues = validate_generated_content(prepared, generated)
        if not issues or retries >= MAX_RETRIES:
            return response

        user_prompt = _retry_prompt(original_prompt, issues)
        retries += 1
