"""Conservative lexical checks for unsupported numeric and commercial claims.

These checks are only heuristics: they compare output with supplied text, not
with independently verified product documentation or semantic entailment.
"""

import re
from decimal import Decimal

from app.preprocessor import PreparedProduct
from app.schemas import LLMGeneratedContent

_NUMBER_RE = re.compile(r"(?<!\w)-?\d+(?:[.,]\d+)?(?!\w)")
_QUANTITY_RE = re.compile(
    r"(?<!\w)(-?\d+(?:[.,]\d+)?)\s*"
    r"(мпа|mpa|кпа|kpa|мв|mv|ма|ma|квт|kw|вт|w|кв|kv|"
    r"в|v|а|a|гц|hz|кгц|khz|°c|°f|мм|mm|см|cm|кг|kg|%)"
    r"(?!\w)",
    re.IGNORECASE,
)
_IP_RE = re.compile(r"(?<!\w)IP\s*\d{2,3}(?!\w)", re.IGNORECASE)

_UNIT_ALIASES = {
    "mpa": "мпа",
    "kpa": "кпа",
    "mv": "мв",
    "ma": "ма",
    "kw": "квт",
    "w": "вт",
    "kv": "кв",
    "v": "в",
    "a": "а",
    "hz": "гц",
    "khz": "кгц",
    "mm": "мм",
    "cm": "см",
    "kg": "кг",
}

_CLAIMS = (
    ("гарантии", re.compile(r"\bгарант\w*", re.IGNORECASE)),
    (
        "сертификации",
        re.compile(r"\b(?:сертификат|сертифицир)\w*", re.IGNORECASE),
    ),
    ("наличии", re.compile(r"\bв\s+наличии\b", re.IGNORECASE)),
    ("доставке", re.compile(r"\bдоставк\w*", re.IGNORECASE)),
    ("скидке", re.compile(r"\bскидк\w*", re.IGNORECASE)),
    (
        "цене",
        re.compile(r"\b(?:цен|стоимост)\w*", re.IGNORECASE),
    ),
)


def _source_text(prepared: PreparedProduct) -> str:
    product = prepared.product
    fields = [
        product.name,
        product.manufacturer or "",
        product.description,
        *[f"{key}: {value}" for key, value in product.specs.items()],
    ]
    return "\n".join(fields)


def _output_text(content: LLMGeneratedContent) -> str:
    return "\n".join(
        [
            content.seo_title,
            content.seo_description,
            *content.search_queries,
            content.commercial_block,
        ]
    )


def _numbers(text: str) -> set[Decimal]:
    return {
        Decimal(match.group().replace(",", ".")) for match in _NUMBER_RE.finditer(text)
    }


def _quantities(text: str) -> set[tuple[Decimal, str]]:
    quantities: set[tuple[Decimal, str]] = set()
    for match in _QUANTITY_RE.finditer(text):
        amount = Decimal(match.group(1).replace(",", "."))
        unit = match.group(2).casefold()
        quantities.add((amount, _UNIT_ALIASES.get(unit, unit)))
    return quantities


def validate_grounding(
    prepared: PreparedProduct, content: LLMGeneratedContent
) -> list[str]:
    """Flag ungrounded quantities, numerals and high-risk commercial topics.

    A matching number or keyword is evidence of mention, not proof of a true
    or semantically identical claim. False positives and misses are possible.
    """
    source = _source_text(prepared)
    generated = _output_text(content)
    warnings: list[str] = []

    source_quantities = _quantities(source)
    output_quantities = _quantities(generated)
    for amount, unit in sorted(output_quantities - source_quantities):
        warnings.append(f"Значение {amount} {unit} не подтверждено исходными данными.")

    paired_numbers = {amount for amount, _ in output_quantities}
    missing_numbers = _numbers(generated) - _numbers(source) - paired_numbers
    for number in sorted(missing_numbers):
        warnings.append(f"Число {number} отсутствует в исходных данных.")

    source_ip = {
        match.group().upper().replace(" ", "") for match in _IP_RE.finditer(source)
    }
    output_ip = {
        match.group().upper().replace(" ", "") for match in _IP_RE.finditer(generated)
    }
    for rating in sorted(output_ip - source_ip):
        warnings.append(f"Класс защиты {rating} не подтверждён исходными данными.")

    for label, pattern in _CLAIMS:
        if pattern.search(generated) and not pattern.search(source):
            warnings.append(f"Утверждение о {label} не подтверждено исходными данными.")

    return warnings
