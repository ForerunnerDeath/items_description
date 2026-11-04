"""Conservative detection of obvious role-spoofing and instruction attempts.

This is a triage filter, not a reliable prompt-injection detector. Attackers
can paraphrase or encode instructions; legitimate text may also be flagged.
"""

import re

_INSTRUCTION_PATTERNS = (
    re.compile(r"(?im)^\s*(?:system|developer|assistant)\s*:"),
    re.compile(r"(?i)\[im_start\]\s*(?:system|developer|assistant)\b"),
    re.compile(r"(?i)<\|im_start\|>\s*(?:system|developer|assistant)\b"),
    re.compile(r"(?i)</?\s*(?:system|developer|assistant)\s*>"),
    re.compile(
        r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?"
        r"(?:previous|prior|system|developer)\s+"
        r"(?:instructions|rules|messages)\b"
    ),
    re.compile(
        r"(?i)\bигнорируй(?:те)?\b[^\n.!?]{0,80}"
        r"\b(?:инструкц\w*|правил\w*|промпт\w*|указан\w*)"
    ),
    re.compile(
        r"(?i)\b(?:покажи|выведи|раскрой)\b[^\n.!?]{0,80}"
        r"\b(?:системн\w*\s+(?:промпт\w*|инструкц\w*)|api[_ -]?key)\b"
    ),
)


def contains_instruction_markers(value: str) -> bool:
    """Identify some explicit instructions masquerading as product content."""
    return any(pattern.search(value) is not None for pattern in _INSTRUCTION_PATTERNS)
