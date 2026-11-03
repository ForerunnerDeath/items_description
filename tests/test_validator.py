"""Business checks for SEO fields and application-owned specifications."""

import pytest

from app.preprocessor import prepare_product
from app.schemas import LLMGeneratedContent, ProductInput
from app.validator import build_generation_response, validate_seo_content


def make_content(**changes: object) -> LLMGeneratedContent:
    fields: dict[str, object] = {
        "seo_title": "БП15Б-Д2-24 — блок питания",
        "seo_description": "А" * 800,
        "search_queries": [
            "купить БП15Б-Д2-24",
            "блок питания ОВЕН",
            "БП15Б-Д2-24 24 В",
        ],
        "commercial_block": "Для задач, описанных в карточке товара.",
    }
    fields.update(changes)
    return LLMGeneratedContent.model_validate(fields)


def test_boundary_lengths_and_three_queries_are_accepted() -> None:
    content = make_content(
        seo_title="А" * 70,
        seo_description="Б" * 800,
        search_queries=["запрос один", "запрос два", "запрос три"],
    )

    assert validate_seo_content(content) == []


def test_upper_description_bound_and_five_queries_are_accepted() -> None:
    content = make_content(
        seo_description="Б" * 1200,
        search_queries=["один", "два", "три", "четыре", "пять"],
    )

    assert validate_seo_content(content) == []


@pytest.mark.parametrize("title", ["", "А" * 71])
def test_invalid_title_length_requires_review(title: str) -> None:
    assert "SEO-заголовок" in " ".join(
        validate_seo_content(make_content(seo_title=title))
    )


@pytest.mark.parametrize("size", [0, 799, 1201])
def test_invalid_description_length_requires_review(size: int) -> None:
    assert "SEO-описание" in " ".join(
        validate_seo_content(make_content(seo_description="А" * size))
    )


@pytest.mark.parametrize(
    "queries",
    [
        [],
        ["один", "два"],
        ["один", "два", "три", "четыре", "пять", "шесть"],
    ],
)
def test_invalid_search_query_count_requires_review(queries: list[str]) -> None:
    assert "от 3 до 5" in " ".join(
        validate_seo_content(make_content(search_queries=queries))
    )


def test_queries_are_unique_ignoring_case_and_extra_spaces() -> None:
    content = make_content(
        search_queries=["  Блок  питания  ОВЕН ", "блок питания овен", "другой"]
    )

    assert "уникальными" in " ".join(validate_seo_content(content))


@pytest.mark.parametrize("empty", ["", " ", "  \n  "])
def test_blank_query_requires_review(empty: str) -> None:
    content = make_content(search_queries=["первый", empty, "третий"])

    assert "не должны быть пустыми" in " ".join(validate_seo_content(content))


def test_empty_commercial_block_requires_review() -> None:
    content = make_content(commercial_block="")

    assert "Коммерческий блок" in " ".join(validate_seo_content(content))


def test_valid_result_copies_specs_without_llm_modification() -> None:
    product = ProductInput(
        name="БП15Б-Д2-24",
        manufacturer="ОВЕН",
        specs={"Напряжение": "24 В", "Точность": "данных нет"},
        description="Блок питания",
    )
    prepared = prepare_product(product)
    content = make_content()

    response = build_generation_response(prepared, content)

    assert response.status == "generated"
    assert response.characteristics == {"Напряжение": "24 В"}
    assert response.missing_fields == ["specs.Точность"]
    assert response.warnings == []

    response.characteristics["Напряжение"] = "999 В"
    response.search_queries.append("добавили")
    assert prepared.product.specs == {"Напряжение": "24 В"}
    assert content.search_queries == [
        "купить БП15Б-Д2-24",
        "блок питания ОВЕН",
        "БП15Б-Д2-24 24 В",
    ]


def test_invalid_seo_result_is_returned_for_review_without_silent_fixing() -> None:
    prepared = prepare_product(
        ProductInput(name="Прибор", specs={"Напряжение": "12 В"})
    )
    content = make_content(seo_title="Т" * 71, seo_description="кратко")

    response = build_generation_response(prepared, content)

    assert response.status == "needs_review"
    assert response.seo_title == "Т" * 71
    assert response.seo_description == "кратко"
    assert len(response.warnings) == 2
    assert response.characteristics == {"Напряжение": "12 В"}
    assert response.missing_fields == ["manufacturer", "description"]


def test_preparation_warning_is_preserved_and_marks_result_for_review() -> None:
    prepared = prepare_product(
        ProductInput(name="Прибор", specs={" ": "невалидная пара", "Ток": "1 А"})
    )

    response = build_generation_response(prepared, make_content())

    assert response.status == "needs_review"
    assert response.warnings == ["Характеристика с пустым названием пропущена."]
    assert response.characteristics == {"Ток": "1 А"}


def test_sparse_product_returns_insufficient_data_without_llm() -> None:
    prepared = prepare_product(ProductInput(name="АНТ-1", manufacturer="ОВЕН"))

    response = build_generation_response(prepared)

    assert response.status == "insufficient_data"
    assert response.seo_title is None
    assert response.characteristics == {}
    assert response.missing_fields == ["description", "specs"]
    assert response.warnings == [
        "Недостаточно данных для генерации: нужны характеристики или описание."
    ]


def test_missing_llm_content_for_ready_product_is_rejected() -> None:
    prepared = prepare_product(
        ProductInput(name="Прибор", description="Измеряет температуру.")
    )

    with pytest.raises(ValueError, match="required"):
        build_generation_response(prepared)


def test_llm_output_for_insufficient_product_is_rejected() -> None:
    prepared = prepare_product(ProductInput(name="АНТ-1"))

    with pytest.raises(ValueError, match="must not exist"):
        build_generation_response(prepared, make_content())
