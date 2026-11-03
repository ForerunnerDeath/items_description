import json

import pytest
from pydantic import ValidationError

from app.schemas import LLMGeneratedContent


VALID_OUTPUT: dict[str, object] = {
    "seo_title": "БП15Б-Д2-24 — блок питания ОВЕН",
    "seo_description": "Описание блока питания на основе исходных данных.",
    "search_queries": ["ОВЕН БП15Б-Д2-24", "блок питания 24 В"],
    "commercial_block": "Для задач, указанных в описании товара.",
}


def test_valid_json_is_parsed_and_preserves_russian_content() -> None:
    result = LLMGeneratedContent.model_validate_json(
        json.dumps(VALID_OUTPUT, ensure_ascii=False)
    )

    assert result.model_dump() == VALID_OUTPUT


def test_json_schema_requires_only_the_four_llm_fields() -> None:
    schema = LLMGeneratedContent.model_json_schema()

    assert set(schema["required"]) == set(VALID_OUTPUT)
    assert set(schema["properties"]) == set(VALID_OUTPUT)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["search_queries"]["type"] == "array"
    assert schema["properties"]["search_queries"]["items"] == {"type": "string"}


@pytest.mark.parametrize("field", list(VALID_OUTPUT))
def test_missing_required_fields_are_rejected(field: str) -> None:
    payload = VALID_OUTPUT.copy()
    payload.pop(field)

    with pytest.raises(ValidationError):
        LLMGeneratedContent.model_validate(payload)


def test_extra_llm_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMGeneratedContent.model_validate(
            {**VALID_OUTPUT, "characteristics": {"Напряжение": "24 В"}}
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("seo_title", 123),
        ("seo_description", None),
        ("search_queries", "один запрос"),
        ("search_queries", ["корректный запрос", 123]),
        ("commercial_block", {"text": "Значение"}),
    ],
)
def test_invalid_field_types_are_rejected(field: str, invalid_value: object) -> None:
    with pytest.raises(ValidationError):
        LLMGeneratedContent.model_validate({**VALID_OUTPUT, field: invalid_value})


def test_surrounding_whitespace_is_trimmed() -> None:
    result = LLMGeneratedContent.model_validate(
        {
            **VALID_OUTPUT,
            "seo_title": "  Название  ",
            "search_queries": ["  запрос один  ", "запрос два  "],
        }
    )

    assert result.seo_title == "Название"
    assert result.search_queries == ["запрос один", "запрос два"]


def test_content_quality_constraints_are_reserved_for_later_validator() -> None:
    result = LLMGeneratedContent(
        seo_title="",
        seo_description="",
        search_queries=[],
        commercial_block="",
    )

    assert result.search_queries == []
