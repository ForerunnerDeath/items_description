"""Tests for Pydantic-backed structured LLM generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError
from pydantic import ValidationError

from app.llm_client import LLMClientError, OpenAICompatibleLLMClient
from app.schemas import LLMGeneratedContent


def make_client(sdk: MagicMock) -> OpenAICompatibleLLMClient:
    with patch("app.llm_client.AsyncOpenAI", return_value=sdk):
        return OpenAICompatibleLLMClient(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            max_tokens=2048,
            timeout_seconds=30.0,
        )


def make_sdk(
    *,
    parsed: LLMGeneratedContent | None = None,
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    response = MagicMock()
    response.choices = [
        MagicMock(
            finish_reason=finish_reason,
            message=MagicMock(parsed=parsed, refusal=refusal),
        )
    ]

    sdk = MagicMock()
    sdk.chat.completions.parse = AsyncMock(return_value=response)
    sdk.close = AsyncMock()
    return sdk


@pytest.mark.asyncio
async def test_structured_call_passes_pydantic_schema_and_returns_model() -> None:
    expected = LLMGeneratedContent(
        seo_title="БП15Б-Д2-24 — блок питания ОВЕН",
        seo_description="Описание блока питания.",
        search_queries=["купить БП15Б-Д2-24", "ОВЕН блок питания"],
        commercial_block="Для задач промышленной автоматизации.",
    )
    sdk = make_sdk(parsed=expected)
    client = make_client(sdk)

    result = await client.generate_structured(
        system_prompt="Rules",
        user_prompt="Product JSON",
    )

    assert result is expected
    sdk.chat.completions.parse.assert_awaited_once_with(
        model="test-model",
        messages=[
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Product JSON"},
        ],
        response_format=LLMGeneratedContent,
        temperature=0,
        max_tokens=2048,
    )


@pytest.mark.asyncio
async def test_structured_call_rejects_refusal() -> None:
    client = make_client(make_sdk(refusal="Не могу ответить."))

    with pytest.raises(LLMClientError) as error:
        await client.generate_structured(system_prompt="Rules", user_prompt="Data")

    assert error.value.error_type == "Refusal"


@pytest.mark.asyncio
async def test_structured_call_rejects_unparsed_response() -> None:
    client = make_client(make_sdk(parsed=None))

    with pytest.raises(LLMClientError) as error:
        await client.generate_structured(system_prompt="Rules", user_prompt="Data")

    assert error.value.error_type == "EmptyResponse"


@pytest.mark.asyncio
async def test_structured_call_rejects_empty_choices() -> None:
    sdk = make_sdk()
    sdk.chat.completions.parse.return_value.choices = []
    client = make_client(sdk)

    with pytest.raises(LLMClientError) as error:
        await client.generate_structured(system_prompt="Rules", user_prompt="Data")

    assert error.value.error_type == "EmptyResponse"


@pytest.mark.parametrize(
    ("finish_reason", "error_type"),
    [
        ("length", "IncompleteResponse"),
        ("content_filter", "FilteredResponse"),
    ],
)
@pytest.mark.asyncio
async def test_structured_call_rejects_incomplete_output(
    finish_reason: str, error_type: str
) -> None:
    client = make_client(make_sdk(finish_reason=finish_reason))

    with pytest.raises(LLMClientError) as error:
        await client.generate_structured(system_prompt="Rules", user_prompt="Data")

    assert error.value.error_type == error_type


@pytest.mark.asyncio
async def test_structured_call_wraps_schema_validation_errors() -> None:
    sdk = make_sdk()
    invalid = {"seo_title": 123}
    validation_error: ValidationError | None = None

    try:
        LLMGeneratedContent.model_validate(invalid)
    except ValidationError as exc:
        validation_error = exc

    assert validation_error is not None
    sdk.chat.completions.parse = AsyncMock(side_effect=validation_error)
    client = make_client(sdk)

    with pytest.raises(LLMClientError) as error:
        await client.generate_structured(system_prompt="Rules", user_prompt="Data")

    assert error.value.error_type == "InvalidResponse"


@pytest.mark.asyncio
async def test_structured_call_wraps_provider_errors() -> None:
    sdk = make_sdk()
    sdk.chat.completions.parse = AsyncMock(
        side_effect=OpenAIError("provider unavailable")
    )
    client = make_client(sdk)

    with pytest.raises(LLMClientError) as error:
        await client.generate_structured(system_prompt="Rules", user_prompt="Data")

    assert error.value.error_type == "OpenAIError"
