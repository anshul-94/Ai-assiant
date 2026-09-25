"""
Text-to-speech module for Qreels AI tutor.
Uses Edge TTS with dynamic voice selection based on detected language.
"""

import asyncio
import re
import edge_tts
import os

# ---------------------------------------------------------------------------
# Voice mapping
# ---------------------------------------------------------------------------
VOICES = {
    "english":  "en-IN-NeerjaNeural",    # Indian English female, friendly teacher
    "hinglish": "hi-IN-SwaraNeural",     # Hindi female, natural for Hinglish
}

# Fallback voices (if primary unavailable)
VOICES_ALT = {
    "english":  "en-IN-PrabhatNeural",
    "hinglish": "hi-IN-MadhurNeural",
}

OUTPUT_FILE = "audio/output.mp3"

# ---------------------------------------------------------------------------
# TTS preparation helpers
# ---------------------------------------------------------------------------

def _clean_for_tts(text: str) -> str:
    """
    Prepare text for TTS synthesis:
    - Strip markdown formatting
    - Remove emojis (TTS mispronounces them)
    - Strip Devanagari (should not be present but guard anyway)
    - Normalise whitespace
    """
    # Remove markdown bold/italic/code
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`[^`\n]+`", "", text)
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove emojis (Unicode emoji ranges)
    text = re.sub(
        r"[\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0"
        r"\U000024C2-\U0001F251"
        r"\u2600-\u26FF\u2700-\u27BF]+",
        " ", text, flags=re.UNICODE
    )

    # Strip Devanagari
    text = re.sub(r"[\u0900-\u097F]+", " ", text)

    # Strip leftover symbols not useful for TTS
    text = re.sub(r"[#\[\]{}|\\^~<>]+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


async def _speak_async(text: str, language: str = "english") -> None:
    """Generate MP3 with Edge TTS and save to audio/output.mp3."""
    clean = _clean_for_tts(text)

    if len(clean.strip()) < 3:
        return

    os.makedirs("audio", exist_ok=True)

    voice = VOICES.get(language, VOICES["english"])

    try:
        communicate = edge_tts.Communicate(
            text=clean,
            voice=voice,
            rate="-5%",    # slightly slower — clearer for children
            pitch="+0Hz",
        )
        await communicate.save(OUTPUT_FILE)
    except Exception as exc:
        # Try fallback voice
        fallback = VOICES_ALT.get(language, VOICES_ALT["english"])
        print(f"TTS primary voice failed ({exc}), trying fallback: {fallback}")
        communicate = edge_tts.Communicate(
            text=clean,
            voice=fallback,
            rate="-5%",
            pitch="+0Hz",
        )
        await communicate.save(OUTPUT_FILE)


def speak(text: str, language: str = "english") -> None:
    """Synchronous wrapper for _speak_async."""
    asyncio.run(_speak_async(text, language))


async def speak_async(text: str, language: str = "english") -> None:
    """Async entry point (used by main.py directly)."""
    await _speak_async(text, language)
