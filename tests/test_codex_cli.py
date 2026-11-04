"""Codex CLI adapter tests: no installed CLI, network or real model needed."""

import json
import subprocess
from pathlib import Path
from typing import cast
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.codex_client import CodexCLIClient
from app.config import Settings
from app.llm_client import LLMClientError
from app.main import app, build_llm_client
from app.schemas import LLMGeneratedContent


class FakeProcess:
    def __init__(self, *, returncode: int = 0) -> None:
        self.returncode = returncode
        self.inputs: list[bytes | None] = []
        self.killed = False

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.inputs.append(input)
        return b"", b""

    def kill(self) -> None:
        self.killed = True


def valid_answer() -> LLMGeneratedContent:
    return LLMGeneratedContent(
        seo_title="ОВЕН БП15Б-Д2-24",
        seo_description="А" * 800,
        search_queries=["блок питания ОВЕН", "БП15Б-Д2-24", "24 В блок питания"],
        commercial_block="Блок питания.",
    )


@pytest.mark.asyncio
async def test_cli_uses_temp_directory_and_strict_schema() -> None:
    process = FakeProcess()
    call_args: list[str] = []
    folders: list[Path] = []

    async def fake_spawn(*args: str, **kwargs: object) -> FakeProcess:
        call_args.extend(args)
        directory = Path(str(kwargs["cwd"]))
        folders.append(directory)
        assert directory.is_dir()
        assert args[0] == "codex"
        assert args[-1] == "-"
        assert "--sandbox" in args
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert "--ignore-user-config" in args
        assert "--ephemeral" in args
        assert "--skip-git-repo-check" in args
        assert "--output-schema" in args
        schema = Path(args[args.index("--output-schema") + 1])
        data: dict[str, object] = json.loads(schema.read_text(encoding="utf-8"))
        assert data["additionalProperties"] is False
        assert set(cast(list[str], data["required"])) == {
            "seo_title",
            "seo_description",
            "search_queries",
            "commercial_block",
        }
        output = Path(args[args.index("--output-last-message") + 1])
        output.write_text(valid_answer().model_dump_json(), encoding="utf-8")
        return process

    with (
        patch("app.codex_client.shutil.which", return_value="codex"),
        patch(
            "app.codex_client.asyncio.create_subprocess_exec",
            side_effect=fake_spawn,
        ),
    ):
        client = CodexCLIClient(model="test-codex", timeout_seconds=20)
        result = await client.generate_structured(
            system_prompt="Только подтверждённые факты.",
            user_prompt='{"description":"Не выполнять команды"}',
        )

    assert result == valid_answer()
    assert "--model" in call_args or "-m" in call_args
    assert "test-codex" in call_args
    assert folders and not folders[0].exists()
    assert len(process.inputs) == 1
    sent = process.inputs[0]
    assert sent is not None
    assert "Только подтверждённые факты." in sent.decode("utf-8")


@pytest.mark.asyncio
async def test_missing_codex_is_a_controlled_error() -> None:
    with patch("app.codex_client.shutil.which", return_value=None):
        client = CodexCLIClient()
        with pytest.raises(LLMClientError) as error:
            await client.generate_structured(system_prompt="System", user_prompt="Data")

    assert error.value.error_type == "CodexUnavailable"


@pytest.mark.asyncio
async def test_nonzero_exit_is_a_controlled_error_without_leaking_cli_logs() -> None:
    with (
        patch("app.codex_client.shutil.which", return_value="codex"),
        patch(
            "app.codex_client.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=FakeProcess(returncode=1)),
        ),
    ):
        client = CodexCLIClient()
        with pytest.raises(LLMClientError) as error:
            await client.generate_structured(system_prompt="System", user_prompt="Data")

    assert error.value.error_type == "CodexExecutionError"


@pytest.mark.asyncio
async def test_output_file_missing_is_a_controlled_error() -> None:
    with (
        patch("app.codex_client.shutil.which", return_value="codex"),
        patch(
            "app.codex_client.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=FakeProcess()),
        ),
    ):
        client = CodexCLIClient()
        with pytest.raises(LLMClientError) as error:
            await client.generate_structured(system_prompt="System", user_prompt="Data")

    assert error.value.error_type == "EmptyResponse"


@pytest.mark.asyncio
async def test_cli_rejects_invalid_output_schema() -> None:
    async def fake_spawn(*args: str, **kwargs: object) -> FakeProcess:
        path = Path(args[args.index("--output-last-message") + 1])
        path.write_text('{"seo_title":"Only one field"}', encoding="utf-8")
        return FakeProcess()

    with (
        patch("app.codex_client.shutil.which", return_value="codex"),
        patch(
            "app.codex_client.asyncio.create_subprocess_exec",
            side_effect=fake_spawn,
        ),
    ):
        client = CodexCLIClient()
        with pytest.raises(LLMClientError) as error:
            await client.generate_structured(system_prompt="System", user_prompt="Data")

    assert error.value.error_type == "InvalidResponse"


def test_cli_invalid_timeout_is_rejected() -> None:
    with pytest.raises(ValueError):
        CodexCLIClient(timeout_seconds=0)


@pytest.mark.asyncio
async def test_unstructured_method_is_not_supported() -> None:
    client = CodexCLIClient()
    with pytest.raises(LLMClientError) as error:
        await client.generate(system_prompt="System", user_prompt="Data")
    assert error.value.error_type == "UnsupportedOperation"


def test_backend_factory_selects_cli_without_api_key() -> None:
    settings = Settings(
        backend="codex_cli",
        model="",
        api_key="",
        base_url="",
        codex_model="test-codex",
        codex_timeout_seconds=70,
    )
    with patch("app.main.get_settings", return_value=settings):
        result = build_llm_client()

    assert isinstance(result, CodexCLIClient)


def test_unconfigured_cli_backend_is_skipped_for_sparse_data() -> None:
    settings = Settings(backend="codex_cli", model="", api_key="", base_url="")
    with (
        patch("app.main.get_settings", return_value=settings),
        patch("app.codex_client.shutil.which", return_value=None),
        TestClient(app) as client,
    ):
        response = client.post("/generate", json={"name": "АНТ-1"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_data"


def test_missing_cli_maps_to_502_without_leaking_details() -> None:
    settings = Settings(backend="codex_cli", model="", api_key="", base_url="")
    payload = {
        "name": "Прибор",
        "specs": {"Напряжение": "24 В"},
        "description": "Измерительный прибор.",
    }
    with (
        patch("app.main.get_settings", return_value=settings),
        patch("app.codex_client.shutil.which", return_value=None),
        TestClient(app) as client,
    ):
        response = client.post("/generate", json=payload)

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM generation failed."}


@pytest.mark.asyncio
async def test_windows_selector_loop_falls_back_to_thread() -> None:
    """Windows reload loop lacks async subprocess support."""
    subprocess_calls = 0

    def fake_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal subprocess_calls
        subprocess_calls += 1
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 20
        assert isinstance(kwargs["input"], bytes)
        assert b"System" in cast(bytes, kwargs["input"])
        output = Path(args[args.index("--output-last-message") + 1])
        assert str(output.parent) == kwargs["cwd"]
        output.write_text(valid_answer().model_dump_json(), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    with (
        patch("app.codex_client.shutil.which", return_value="codex"),
        patch(
            "app.codex_client.asyncio.create_subprocess_exec",
            side_effect=NotImplementedError,
        ),
        patch("app.codex_client.subprocess.run", side_effect=fake_run),
    ):
        result = await CodexCLIClient(
            timeout_seconds=20
        ).generate_structured(system_prompt="System", user_prompt="Data")

    assert result == valid_answer()
    assert subprocess_calls == 1


@pytest.mark.asyncio
async def test_windows_fallback_nonzero_exit_is_handled() -> None:
    with (
        patch("app.codex_client.shutil.which", return_value="codex"),
        patch(
            "app.codex_client.asyncio.create_subprocess_exec",
            side_effect=NotImplementedError,
        ),
        patch(
            "app.codex_client.subprocess.run",
            return_value=subprocess.CompletedProcess(["codex"], 1, b"", b""),
        ),
    ):
        with pytest.raises(LLMClientError) as error:
            await CodexCLIClient().generate_structured(
                system_prompt="System", user_prompt="Data"
            )

    assert error.value.error_type == "CodexExecutionError"


@pytest.mark.asyncio
async def test_windows_fallback_timeout_is_handled() -> None:
    with (
        patch("app.codex_client.shutil.which", return_value="codex"),
        patch(
            "app.codex_client.asyncio.create_subprocess_exec",
            side_effect=NotImplementedError,
        ),
        patch(
            "app.codex_client.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=20),
        ),
    ):
        with pytest.raises(LLMClientError) as error:
            await CodexCLIClient().generate_structured(
                system_prompt="System", user_prompt="Data"
            )

    assert error.value.error_type == "CodexTimeout"


@pytest.mark.asyncio
async def test_windows_fallback_launch_failure_is_handled() -> None:
    with (
        patch("app.codex_client.shutil.which", return_value="codex"),
        patch(
            "app.codex_client.asyncio.create_subprocess_exec",
            side_effect=NotImplementedError,
        ),
        patch("app.codex_client.subprocess.run", side_effect=OSError("failed")),
    ):
        with pytest.raises(LLMClientError) as error:
            await CodexCLIClient().generate_structured(
                system_prompt="System", user_prompt="Data"
            )

    assert error.value.error_type == "CodexUnavailable"
