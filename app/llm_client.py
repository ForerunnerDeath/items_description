"""Minimal async client for OpenAI-compatible chat-completions APIs."""

from typing import Protocol

from openai import AsyncOpenAI, OpenAIError


class LLMClientError(Exception):
    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class LLMClient(Protocol):
    async def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...

    async def close(self) -> None: ...


class OpenAICompatibleLLMClient:
    """Transport boundary; prompt building and JSON validation live elsewhere."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        if not model.strip():
            raise ValueError("LLM model must not be empty")
        if max_tokens <= 0:
            raise ValueError("LLM max_tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("LLM timeout_seconds must be positive")
        if not api_key.strip() and not base_url.strip():
            raise ValueError("LLM API key is required for the default OpenAI endpoint")

        self._model = model.strip()
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url or None,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=self._max_tokens,
            )
        except OpenAIError as exc:
            raise LLMClientError(str(exc), error_type=type(exc).__name__) from exc

        if not response.choices:
            raise LLMClientError("LLM returned no choices", error_type="EmptyResponse")

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMClientError(
                "LLM response was truncated by the token limit",
                error_type="IncompleteResponse",
            )

        content = choice.message.content
        if not content or not content.strip():
            raise LLMClientError("LLM returned empty content", error_type="EmptyResponse")

        return content.strip()

    async def close(self) -> None:
        await self._client.close()
