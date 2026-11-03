from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

from app.llm_client import LLMClientError, OpenAICompatibleLLMClient


def make_sdk_mock(*, content: str = "  Готовое описание.  ") -> MagicMock:
    response = MagicMock()
    response.choices = [
        MagicMock(
            finish_reason="stop",
            message=MagicMock(content=content),
        )
    ]
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(return_value=response)
    sdk.close = AsyncMock()
    return sdk


@pytest.mark.asyncio
async def test_generate_passes_prompts_and_returns_stripped_text() -> None:
    sdk = make_sdk_mock()

    with patch("app.llm_client.AsyncOpenAI", return_value=sdk) as constructor:
        client = OpenAICompatibleLLMClient(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            max_tokens=2048,
            timeout_seconds=30.0,
        )
        result = await client.generate(
            system_prompt="Rules",
            user_prompt="Product data",
        )

    assert result == "Готовое описание."
    constructor.assert_called_once_with(
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        timeout=30.0,
        max_retries=0,
    )
    sdk.chat.completions.create.assert_awaited_once_with(
        model="test-model",
        messages=[
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Product data"},
        ],
        temperature=0,
        max_tokens=2048,
    )


@pytest.mark.asyncio
async def test_openai_compatible_local_backend_accepts_empty_key() -> None:
    sdk = make_sdk_mock()

    with patch("app.llm_client.AsyncOpenAI", return_value=sdk) as constructor:
        client = OpenAICompatibleLLMClient(
            model="local-model",
            api_key="",
            base_url="http://localhost:11434/v1",
            max_tokens=512,
            timeout_seconds=10,
        )
        await client.close()

    assert constructor.call_args.kwargs["api_key"] == "not-needed"
    sdk.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_sdk_exception_is_wrapped() -> None:
    sdk = make_sdk_mock()
    sdk.chat.completions.create = AsyncMock(side_effect=OpenAIError("LLM failed"))

    with patch("app.llm_client.AsyncOpenAI", return_value=sdk):
        client = OpenAICompatibleLLMClient(
            model="model",
            api_key="key",
            base_url="",
            max_tokens=512,
            timeout_seconds=10,
        )
        with pytest.raises(LLMClientError) as exc_info:
            await client.generate(system_prompt="Rules", user_prompt="Data")

    assert exc_info.value.error_type == "OpenAIError"
    assert "LLM failed" in str(exc_info.value)


@pytest.mark.parametrize("text", ["", "   "])
@pytest.mark.asyncio
async def test_empty_llm_text_is_reported_as_error(text: str) -> None:
    sdk = make_sdk_mock(content=text)

    with patch("app.llm_client.AsyncOpenAI", return_value=sdk):
        client = OpenAICompatibleLLMClient(
            model="model",
            api_key="key",
            base_url="",
            max_tokens=512,
            timeout_seconds=10,
        )
        with pytest.raises(LLMClientError) as exc_info:
            await client.generate(system_prompt="Rules", user_prompt="Data")

    assert exc_info.value.error_type == "EmptyResponse"


@pytest.mark.asyncio
async def test_no_choices_is_reported_as_error() -> None:
    sdk = make_sdk_mock()
    sdk.chat.completions.create.return_value.choices = []

    with patch("app.llm_client.AsyncOpenAI", return_value=sdk):
        client = OpenAICompatibleLLMClient(
            model="model",
            api_key="key",
            base_url="",
            max_tokens=512,
            timeout_seconds=10,
        )
        with pytest.raises(LLMClientError) as exc_info:
            await client.generate(system_prompt="Rules", user_prompt="Data")

    assert exc_info.value.error_type == "EmptyResponse"


@pytest.mark.asyncio
async def test_truncated_generation_is_rejected() -> None:
    sdk = make_sdk_mock()
    sdk.chat.completions.create.return_value.choices[0].finish_reason = "length"

    with patch("app.llm_client.AsyncOpenAI", return_value=sdk):
        client = OpenAICompatibleLLMClient(
            model="model",
            api_key="key",
            base_url="",
            max_tokens=512,
            timeout_seconds=10,
        )
        with pytest.raises(LLMClientError) as exc_info:
            await client.generate(system_prompt="Rules", user_prompt="Data")

    assert exc_info.value.error_type == "IncompleteResponse"


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"model": "   "},
        {"max_tokens": 0},
        {"timeout_seconds": 0},
        {"api_key": "", "base_url": ""},
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict[str, object]) -> None:
    options: dict[str, object] = {
        "model": "model",
        "api_key": "key",
        "base_url": "",
        "max_tokens": 512,
        "timeout_seconds": 10.0,
    }
    options.update(overrides)

    with pytest.raises(ValueError):
        OpenAICompatibleLLMClient(
            model=str(options["model"]),
            api_key=str(options["api_key"]),
            base_url=str(options["base_url"]),
            max_tokens=int(options["max_tokens"]),  # pyright: ignore[reportArgumentType]
            timeout_seconds=float(options["timeout_seconds"]),  # pyright: ignore[reportArgumentType]
        )
