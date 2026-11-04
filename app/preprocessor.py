"""Prepare verified product facts before asking an LLM to write about them."""

from dataclasses import dataclass

from app.schemas import ProductInput
from app.security import contains_instruction_markers

_MISSING_VALUES = frozenset(
    {"", "-", "—", "–", "н/д", "нет данных", "данных нет", "не указано", "неизвестно"}
)
_MISSING_PREFIXES = ("данных нет ", "не указано ")


def _is_missing(value: str) -> bool:
    normalized = value.strip().casefold()
    return normalized in _MISSING_VALUES or normalized.startswith(_MISSING_PREFIXES)


_IDENTIFIER_KEYS = frozenset(
    {
        "артикул",
        "код товара",
        "каталожный номер",
        "номер по каталогу",
        "код производителя",
        "sku",
        "ean",
        "штрихкод",
    }
)


def _is_identifier_key(name: str) -> bool:
    """Catalog identifiers identify a product, but are not device properties."""
    normalized = " ".join(name.strip().casefold().replace("ё", "е").split())
    return normalized in _IDENTIFIER_KEYS


@dataclass(frozen=True, slots=True)
class PreparedProduct:
    """Normalized facts and diagnostics; this does not verify their truthfulness."""

    product: ProductInput
    missing_fields: tuple[str, ...]
    warnings: tuple[str, ...]
    can_generate: bool
    blocked: bool = False


def prepare_product(product: ProductInput) -> PreparedProduct:
    """Exclude unknown specifications and check whether textual facts exist.

    A nonempty description or at least one known specification permits an LLM
    attempt; this is only a preliminary data-presence check, not a guarantee
    that an 800–1200 character description can be written faithfully.
    The presence of catalog identifiers alone does not make the input
    technically informative. An identifier-only product with a type
    description can still produce a draft, but it needs human review.
    An identifier-only product without any description cannot be generated.
    """
    missing_fields: list[str] = []
    warnings: list[str] = []
    raw_fields = [product.name, product.manufacturer or "", product.description]
    for key, value in product.specs.items():
        raw_fields.extend((key, value))

    blocked = any(contains_instruction_markers(text) for text in raw_fields)
    if blocked:
        warnings.append(
            "В исходных данных обнаружены признаки управляющих инструкций; "
            "требуется ручная проверка."
        )

    manufacturer = (product.manufacturer or "").strip() or None
    description = product.description.strip()
    known_specs: dict[str, str] = {}

    if manufacturer is None:
        missing_fields.append("manufacturer")

    for raw_name, raw_value in product.specs.items():
        name = raw_name.strip()
        value = raw_value.strip()

        if not name:
            warnings.append("Характеристика с пустым названием пропущена.")
            continue

        if _is_missing(value):
            missing_fields.append(f"specs.{name}")
            continue

        known_specs[name] = value

    if _is_missing(description):
        description = ""
        missing_fields.append("description")

    if not known_specs:
        missing_fields.append("specs")
    elif all(_is_identifier_key(name) for name in known_specs):
        missing_fields.append("technical_specs")
        warnings.append(
            "Указаны только идентификаторы товара: технические характеристики "
            "отсутствуют. Черновик SEO-карточки требует проверки."
        )

    has_technical_specs = any(
        not _is_identifier_key(name) for name in known_specs
    )
    can_generate = bool(has_technical_specs or description)
    if not can_generate:
        warnings.append(
            "Недостаточно данных для генерации: нужны характеристики или описание."
        )

    return PreparedProduct(
        product=ProductInput(
            name=product.name,
            manufacturer=manufacturer,
            specs=known_specs,
            description=description,
        ),
        missing_fields=tuple(missing_fields),
        warnings=tuple(warnings),
        can_generate=can_generate,
        blocked=blocked,
    )
