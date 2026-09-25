"""
Language / style detector for Qreels AI tutor.

Returns one of two categories:
  - "english"   -> pure English question
  - "hinglish"  -> Hindi written in Roman letters, or mixed Hindi-English
"""

import re

# ---------------------------------------------------------------------------
# Hinglish signal words — common Hindi words written in Roman letters
# ---------------------------------------------------------------------------
_HINDI_WORDS = {
    # Question words
    "kya", "kaise", "kaun", "kahan", "kab", "kyun", "kyunki",
    "kitna", "kitne", "kitni",
    # Verbs / helpers
    "hai", "hain", "ho", "hoga", "hogi", "hona", "tha", "thi", "the",
    "kar", "karo", "karna", "karte", "karta", "karti",
    "batao", "bolo", "sikho", "samjhao", "batana", "bolna",
    "chahiye", "milta", "milti",
    "padh", "parhna", "likhna", "suno", "sunao",
    # Numbers in Hindi
    "ek", "do", "teen", "char", "paanch", "chhe", "saat", "aath", "nau", "das",
    # Postpositions / conjunctions
    "ka", "ki", "ke", "ko", "mein", "se", "par", "tak", "aur", "ya", "lekin",
    "nahi", "nahin", "mat", "na",
    # Pronouns
    "mera", "meri", "mere", "tera", "teri", "tere", "uska", "uski",
    "yeh", "woh", "hum", "tum", "aap", "main",
    # Misc
    "accha", "theek", "sahi", "samajh", "gaye", "seekh",
    "table", "ginti", "sawaal", "jawab", "naam",
}

_HINDI_PATTERNS = [
    r"\b\w+ta\s+hai\b",
    r"\b\w+ti\s+hai\b",
    r"\b\w+te\s+hain\b",
    r"\bkya\s+h",
    r"\bkaise\b",
    r"\bkaun\b",
    r"\bkahan\b",
    r"\bkyun\b",
    r"\bbatao\b",
    r"\bbolo\b",
    r"\bsunao\b",
    r"\bka\s+table\b",
    r"\bka\s+naam\b",
    r"\bkya\s+hota\b",
    r"\bke\s+baare\b",
    r"\bkaro\b",
    r"\bpadho\b",
    r"\btum\s+kaun\b",
    r"\baap\s+kaun\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _HINDI_PATTERNS]


def detect_language(text: str) -> str:
    """Returns 'hinglish' or 'english'."""
    if not text or not text.strip():
        return "english"

    # Devanagari characters present
    if re.search(r"[\u0900-\u097F]", text):
        return "hinglish"

    lower = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", lower)
    hindi_hits = sum(1 for t in tokens if t in _HINDI_WORDS)

    if hindi_hits >= 1:
        return "hinglish"

    for pattern in _COMPILED:
        if pattern.search(lower):
            return "hinglish"

    return "english"
