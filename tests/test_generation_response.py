import pytest
from pydantic import ValidationError

from app.schemas import GenerationResponse


def test_generated_response_preserves_facts_and_text() -> None:
    response = GenerationResponse.model_validate(
        {
            "status": "generated",
            "seo_title": "  БП15Б-Д2-24 — блок питания ОВЕН  ",
            "seo_description": "Описание товара.",
            "characteristics": {"Выходное напряжение": "24 В"},
            "search_queries": ["купить БП15Б-Д2-24", "ОВЕН блок питания 24 В"],
            "commercial_block": "Для промышленной автоматизации.",
            "missing_fields": [],
            "warnings": [],
        }
    )

    assert response.status == "generated"
    assert response.seo_title == "БП15Б-Д2-24 — блок питания ОВЕН"
    assert response.characteristics == {"Выходное напряжение": "24 В"}
    assert response.search_queries == [
        "купить БП15Б-Д2-24",
        "ОВЕН блок питания 24 В",
    ]
    assert response.model_dump()["commercial_block"] == (
        "Для промышленной автоматизации."
    )


def test_insufficient_data_response_serializes_empty_fields() -> None:
    response = GenerationResponse(
        status="insufficient_data",
        missing_fields=["specs", "description"],
        warnings=["Недостаточно данных для достоверного описания."],
    )

    assert response.model_dump() == {
        "status": "insufficient_data",
        "seo_title": None,
        "seo_description": None,
        "characteristics": {},
        "search_queries": [],
        "commercial_block": None,
        "missing_fields": ["specs", "description"],
        "warnings": ["Недостаточно данных для достоверного описания."],
    }


def test_needs_review_can_contain_partial_result() -> None:
    response = GenerationResponse(
        status="needs_review",
        seo_title="Пирометр testo 830-T1",
        missing_fields=["точность"],
    )

    assert response.status == "needs_review"
    assert response.seo_title == "Пирометр testo 830-T1"
    assert response.seo_description is None


def test_output_collections_do_not_share_state() -> None:
    first = GenerationResponse(status="insufficient_data")
    second = GenerationResponse(status="insufficient_data")

    first.characteristics["Напряжение"] = "24 В"
    first.search_queries.append("купить прибор")
    first.missing_fields.append("description")
    first.warnings.append("Недостаточно сведений")

    assert second.characteristics == {}
    assert second.search_queries == []
    assert second.missing_fields == []
    assert second.warnings == []


def test_missing_or_invalid_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationResponse.model_validate({})

    with pytest.raises(ValidationError):
        GenerationResponse.model_validate({"status": "success"})


@pytest.mark.parametrize(
    "field, value",
    [
        ("seo_title", 12),
        ("seo_description", []),
        ("characteristics", {"Напряжение": 24}),
        ("search_queries", "купить"),
        ("commercial_block", 42),
        ("missing_fields", [123]),
        ("warnings", [False]),
    ],
)
def test_wrong_output_types_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        GenerationResponse.model_validate({"status": "generated", field: value})


def test_unexpected_output_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationResponse.model_validate({"status": "generated", "rating": 5})
