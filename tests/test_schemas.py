import pytest
from pydantic import ValidationError

from app.schemas import ProductInput


def test_complete_product_is_preserved() -> None:
    product = ProductInput.model_validate(
        {
            "name": "  БП15Б-Д2-24  ",
            "manufacturer": "ОВЕН",
            "specs": {"Выходное напряжение": "24 В", "Выходной ток": "0,63 А"},
            "description": "Одноканальный блок питания.",
        }
    )

    assert product.model_dump() == {
        "name": "БП15Б-Д2-24",
        "manufacturer": "ОВЕН",
        "specs": {"Выходное напряжение": "24 В", "Выходной ток": "0,63 А"},
        "description": "Одноканальный блок питания.",
    }


def test_missing_optional_fields_get_defaults() -> None:
    product = ProductInput.model_validate({"name": "АНТ-1"})

    assert product.manufacturer is None
    assert product.specs == {}
    assert product.description == ""


def test_missing_optional_dicts_do_not_share_mutable_state() -> None:
    first = ProductInput(name="Товар А")
    second = ProductInput(name="Товар Б")

    first.specs["Напряжение"] = "24 В"

    assert second.specs == {}


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_or_whitespace_only_name_is_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        ProductInput.model_validate({"name": name})


def test_missing_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProductInput.model_validate({"description": "Товар без названия"})


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"name": 123},
        {"name": "Прибор", "manufacturer": 123},
        {"name": "Прибор", "description": None},
        {"name": "Прибор", "specs": []},
        {"name": "Прибор", "specs": {"Напряжение": 24}},
    ],
)
def test_wrong_input_types_are_rejected(invalid_data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ProductInput.model_validate(invalid_data)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ProductInput.model_validate({"name": "Прибор", "invented": True})


def test_empty_spec_value_is_allowed_for_later_missing_data_check() -> None:
    product = ProductInput.model_validate(
        {"name": "Прибор", "specs": {"Класс точности": ""}}
    )

    assert product.specs == {"Класс точности": ""}
