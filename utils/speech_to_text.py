"""
Speech-to-text using faster-whisper.
Language is set to None so Whisper auto-detects English and Hindi/Hinglish.
"""

from faster_whisper import WhisperModel

# Load once at import time
# compute_type="int8" is fast and sufficient for kids' speech
model = WhisperModel("small", compute_type="int8")


def transcribe(audio_path: str) -> str:
    """
    Transcribe audio file to text.
    language=None  → Whisper auto-detects (supports English + Hindi).
    """
    segments, info = model.transcribe(
        audio_path,
        language=None,   # auto-detect: handles English and Hindi/Hinglish
        beam_size=5,
    )

    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip()
