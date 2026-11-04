import pytest

from app.preprocessor import prepare_product
from app.schemas import ProductInput


def test_complete_product_is_ready_and_preserves_facts() -> None:
    input_product = ProductInput(
        name="БП15Б-Д2-24",
        manufacturer="ОВЕН",
        specs={"Напряжение": "24 В", "Ток": "0,63 А"},
        description="Блок питания",
    )

    result = prepare_product(input_product)

    assert result.can_generate is True
    assert result.missing_fields == ()
    assert result.warnings == ()
    assert result.product.model_dump() == input_product.model_dump()


def test_sparse_product_is_not_ready() -> None:
    result = prepare_product(ProductInput(name="АНТ-1", manufacturer="ОВЕН"))

    assert result.can_generate is False
    assert result.missing_fields == ("description", "specs")
    assert result.product.specs == {}
    assert result.warnings == (
        "Недостаточно данных для генерации: нужны характеристики или описание.",
    )


def test_description_alone_allows_attempt() -> None:
    result = prepare_product(
        ProductInput(name="Прибор", description="Описание назначения прибора.")
    )

    assert result.can_generate is True
    assert result.missing_fields == ("manufacturer", "specs")


def test_one_spec_alone_allows_attempt() -> None:
    result = prepare_product(ProductInput(name="Прибор", specs={"Напряжение": "24 В"}))

    assert result.can_generate is True
    assert result.missing_fields == ("manufacturer", "description")


def test_missing_marker_for_a_specific_variant_is_not_a_fact() -> None:
    result = prepare_product(
        ProductInput(
            name="ТМ-510Р.00",
            specs={
                "Класс точности": "данных нет для выбранной модификации",
                "Диапазон": "0–2,5 МПа",
            },
        )
    )

    assert result.product.specs == {"Диапазон": "0–2,5 МПа"}
    assert result.missing_fields == (
        "manufacturer",
        "specs.Класс точности",
        "description",
    )
    assert result.can_generate is True


@pytest.mark.parametrize(
    "marker",
    ["", " ", "—", "-", "н/д", "НЕТ ДАННЫХ", "не указано", "неизвестно"],
)
def test_unknown_spec_values_are_removed(marker: str) -> None:
    result = prepare_product(
        ProductInput(name="Прибор", specs={"Класс точности": marker})
    )

    assert result.can_generate is False
    assert result.product.specs == {}
    assert "specs.Класс точности" in result.missing_fields
    assert "specs" in result.missing_fields


def test_meaningful_negative_value_is_preserved() -> None:
    result = prepare_product(ProductInput(name="Прибор", specs={"Наличие": "нет"}))

    assert result.can_generate is True
    assert result.product.specs == {"Наличие": "нет"}


def test_whitespace_is_normalized_without_mutating_input() -> None:
    input_product = ProductInput(
        name="Прибор",
        manufacturer="  ОВЕН  ",
        specs={"  Диапазон  ": "  0–2,5 МПа  "},
        description="  Показывающий манометр.  ",
    )
    original_specs = input_product.specs.copy()

    result = prepare_product(input_product)

    assert result.product.manufacturer == "ОВЕН"
    assert result.product.specs == {"Диапазон": "0–2,5 МПа"}
    assert result.product.description == "Показывающий манометр."
    assert input_product.specs == original_specs


def test_empty_spec_name_is_ignored_with_warning() -> None:
    result = prepare_product(
        ProductInput(name="Прибор", specs={"  ": "24 В", "Напряжение": "12 В"})
    )

    assert result.product.specs == {"Напряжение": "12 В"}
    assert "Характеристика с пустым названием пропущена." in result.warnings


def test_description_placeholder_is_not_treated_as_context() -> None:
    result = prepare_product(ProductInput(name="Прибор", description="Данных нет"))

    assert result.can_generate is False
    assert result.product.description == ""
    assert "description" in result.missing_fields


@pytest.mark.parametrize(
    "description",
    [
        "комбинированный модуль ввода/вывода дискретных сигналов",
        "пирометр, инфракрасный термометр с лазерным целеуказателем",
    ],
)
def test_feed_identifiers_require_review_but_allow_draft(
    description: str,
) -> None:
    product = ProductInput(
        name="Прибор",
        manufacturer="ОВЕН",
        specs={"Артикул": "58953", "Код товара": "63483"},
        description=description,
    )

    result = prepare_product(product)

    assert result.can_generate is True
    assert result.missing_fields == ("technical_specs",)
    assert len(result.warnings) == 1
    assert "технические характеристики" in result.warnings[0]
    assert result.product.specs == product.specs


def test_identifiers_only_without_description_are_insufficient() -> None:
    result = prepare_product(
        ProductInput(
            name="Прибор",
            manufacturer="ОВЕН",
            specs={"Артикул": "58953", "Код товара": "63483"},
        )
    )

    assert result.can_generate is False
    assert result.missing_fields == ("description", "technical_specs")
    assert any("Недостаточно данных" in item for item in result.warnings)


def test_identifier_key_normalization_and_mixed_technical_specs() -> None:
    result = prepare_product(
        ProductInput(
            name="Прибор",
            specs={
                " АРТИКУЛ ": "ABC",
                "  Код   товара ": "123",
                "Диапазон измерений": "-30 … +400 °C",
            },
        )
    )

    assert result.can_generate is True
    assert "technical_specs" not in result.missing_fields
    assert not result.warnings
