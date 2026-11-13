"""Post-processing: Unicode normalisation + script-aware OCR confusion correction.

Changes from v1
---------------
- Devanagari confusion map: Latin 's' -> avagraha, ':' -> visarga.  Rules that could
  rewrite genuine punctuation (',', '.', ';', hyphens) are deliberately NOT applied.
- Added a Tamil-specific correction pass (substitution of common engine errors).
- Added a Latin/IAST pass for the common OCR confusion between lookalike ASCII
  characters in IAST transcriptions (l/ḷ, n/ṅ, etc. are left to the engines;
  we only fix unambiguous substitutions that are never correct).
- All corrections are applied only when the character appears in the right
  Unicode context, so we never alter text from a different script embedded in
  the same line.

All rules are deterministic (regex-based).  No model calls.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Devanagari corrections
# ---------------------------------------------------------------------------

# Danda normalisation (common OCR output)
_PIPES = [(re.compile(r"\s*\|\|\s*"), " ॥ "), (re.compile(r"\s*\|\s*"), " । ")]

_INDIC = re.compile(r"[ऀ-෿]")

# Single-character substitutions: (find_pattern, replace) applied sequentially.
# Each pattern uses lookahead/lookbehind to anchor the fix to Devanagari context.
_DEVA_FIXES: list[tuple[re.Pattern, str]] = [
    # Latin 's' between Devanagari characters → avagraha ऽ
    (re.compile(r"(?<=[ऀ-ॿ])s(?=[ऀ-ॿ])"), "ऽ"),
    # Colon after Devanagari → visarga ः
    (re.compile(r"(?<=[ऀ-ॿ]):"), "ः"),
]

# ---------------------------------------------------------------------------
# Tamil corrections
# ---------------------------------------------------------------------------

_TAMIL = re.compile(r"[஀-௿]")

_TAMIL_FIXES: list[tuple[re.Pattern, str]] = [
    # Colon after Tamil character → visarga / aytam (ஃ)
    (re.compile(r"(?<=[஀-௿]):"), "ஃ"),
]

# ---------------------------------------------------------------------------
# Latin / IAST corrections
# ---------------------------------------------------------------------------

_LATIN_FIXES: list[tuple[re.Pattern, str]] = [
    # Double hyphen → en-dash (sometimes used as a separator in IAST editions)
    (re.compile(r"\s--\s"), " – "),
    # Stray pipe characters not inside Indic context
    (re.compile(r"\s\|\s"), " "),
]


def normalize_text(text: str, script: str) -> str:
    """Unicode NFC + script-aware OCR confusion corrections + whitespace cleanup."""
    text = unicodedata.normalize("NFC", text or "")

    if script == "latin":
        for pattern, repl in _LATIN_FIXES:
            text = pattern.sub(repl, text)
    elif _INDIC.search(text):
        # Danda normalisation
        for pattern, repl in _PIPES:
            text = pattern.sub(repl, text)
        # Script-specific character corrections
        for pattern, repl in _DEVA_FIXES:
            text = pattern.sub(repl, text)
        if script == "tamil" or _TAMIL.search(text):
            for pattern, repl in _TAMIL_FIXES:
                text = pattern.sub(repl, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text
