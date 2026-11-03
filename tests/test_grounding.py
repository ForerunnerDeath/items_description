"""Heuristic checks for numeric claims and high-risk commercial statements."""

import pytest

from app.generator import generate_product_description
from app.grounding import validate_grounding
from app.preprocessor import prepare_product
from app.schemas import LLMGeneratedContent, ProductInput
from app.validator import build_generation_response


def make_content(**updates: object) -> LLMGeneratedContent:
    values: dict[str, object] = {
        "seo_title": "ОВЕН БП15Б-Д2-24",
        "seo_description": "Описание блока питания.",
        "search_queries": [
            "ОВЕН БП15Б-Д2-24",
            "блок питания 24 В",
            "БП15Б-Д2-24",
        ],
        "commercial_block": "Блок питания.",
    }
    values.update(updates)
    return LLMGeneratedContent.model_validate(values)


def source_product(**updates: object) -> ProductInput:
    data: dict[str, object] = {
        "name": "БП15Б-Д2-24",
        "manufacturer": "ОВЕН",
        "specs": {"Напряжение": "24 В", "Класс защиты": "IP54"},
        "description": "Одноканальный блок питания.",
    }
    data.update(updates)
    return ProductInput.model_validate(data)


def test_known_numbers_units_and_model_are_accepted() -> None:
    prepared = prepare_product(source_product())
    content = make_content(
        seo_description="ОВЕН БП15Б-Д2-24, выходное напряжение 24 В, IP54.",
    )

    assert validate_grounding(prepared, content) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Выходное напряжение 48 В.", "48 в"),
        ("Выходной ток 24 А.", "24 а"),
        ("Температура до 100 °C.", "100 °c"),
        ("Срок гарантии 10 лет.", "10"),
        ("Класс защиты IP67.", "IP67"),
    ],
)
def test_unsupported_values_are_flagged(text: str, expected: str) -> None:
    warnings = validate_grounding(
        prepare_product(source_product()), make_content(seo_description=text)
    )

    assert any(expected in warning for warning in warnings)


@pytest.mark.parametrize(
    ("statement", "claim"),
    [
        ("Предоставляется гарантия.", "гарантии"),
        ("Есть сертификаты соответствия.", "сертификации"),
        ("Товар в наличии.", "наличии"),
        ("Быстрая доставка.", "доставке"),
        ("Действует скидка.", "скидке"),
        ("Выгодная цена.", "цене"),
    ],
)
def test_commercial_claim_without_source_is_flagged(statement: str, claim: str) -> None:
    warnings = validate_grounding(
        prepare_product(source_product()),
        make_content(commercial_block=statement),
    )

    assert any(claim in warning for warning in warnings)


def test_commercial_topic_mentioned_in_source_is_not_flagged() -> None:
    source = source_product(description="Гарантия 12 месяцев. Доставка по России.")
    content = make_content(commercial_block="Гарантия 12 месяцев. Доставка по России.")

    assert validate_grounding(prepare_product(source), content) == []


def test_unconfirmed_claim_marks_complete_length_response_for_review() -> None:
    prepared = prepare_product(source_product())
    content = make_content(
        seo_description="А" * 800,
        commercial_block="Товар в наличии.",
    )

    result = build_generation_response(prepared, content)

    assert result.status == "needs_review"
    assert any("наличии" in warning for warning in result.warnings)
    assert result.characteristics == {"Напряжение": "24 В", "Класс защиты": "IP54"}


class StubClient:
    def __init__(self, contents: list[LLMGeneratedContent]) -> None:
        self.contents = contents
        self.prompts: list[str] = []

    async def generate_structured(
        self, *, system_prompt: str, user_prompt: str
    ) -> LLMGeneratedContent:
        self.prompts.append(user_prompt)
        return self.contents.pop(0)

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("Unexpected unstructured LLM request")

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_retries_on_numeric_claim_and_returns_corrected_text() -> None:
    bad = make_content(
        seo_description="Выходное напряжение 48 В." + "А" * 775,
    )
    valid = make_content(seo_description="А" * 800)
    client = StubClient([bad, valid])

    result = await generate_product_description(source_product(), client)

    assert result.status == "generated"
    assert len(client.prompts) == 2
    assert "48 в" in client.prompts[1]
    assert "Выходное напряжение 48 В." not in client.prompts[1]


@pytest.mark.asyncio
async def test_exhausted_grounding_retries_requires_review() -> None:
    bad = make_content(
        seo_description="А" * 800,
        commercial_block="Действует скидка.",
    )
    client = StubClient([bad, bad, bad])

    result = await generate_product_description(source_product(), client)

    assert result.status == "needs_review"
    assert len(client.prompts) == 3
    assert any("скидке" in warning for warning in result.warnings)


def test_heuristic_does_not_guarantee_semantic_truth() -> None:
    source = source_product(description="Доставка не осуществляется.")
    content = make_content(commercial_block="Доставка осуществляется.")

    assert validate_grounding(prepare_product(source), content) == []
