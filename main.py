"""
Qreels AI Tutor — FastAPI Backend  v3.0
- /chat       → text chat with language detection + OpenRouter LLM
- /voice-chat → voice mode: STT → LLM → TTS (language-aware)
- /health     → status check
"""

import os
import asyncio
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Load .env BEFORE any os.getenv calls
load_dotenv()

# Project utils (imported after load_dotenv so they pick up the key)
from utils.language_detector import detect_language
from utils.llm_response import get_answer
from utils.text_to_speech import speak_async

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Qreels AI Tutor", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "openai/gpt-4o-mini"
TTS_OUTPUT_FILE    = "audio/output.mp3"

print(f"OPENROUTER_API_KEY loaded: {bool(OPENROUTER_API_KEY)}")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    language: str       # 'english' or 'hinglish' — useful for frontend/debug

class VoiceChatRequest(BaseModel):
    text: str

class VoiceChatResponse(BaseModel):
    reply: str
    audio_url: str
    language: str

# ---------------------------------------------------------------------------
# OpenRouter LLM helper (used by /chat directly for async HTTP)
# ---------------------------------------------------------------------------

_SYSTEM_ENGLISH = """\
You are Qreels, a friendly AI school tutor for Nursery to Class 5 students.

You MUST answer ONLY in simple, clear English.
Never switch to Hinglish or Hindi.
Never use Devanagari/Hindi Unicode characters.

SCOPE: School subjects — English, maths, science, general knowledge,
poems, animals, shapes, colours, numbers, tables, and homework.

RULES:
1. Short, age-appropriate answers (1-3 sentences for simple questions).
2. For maths, show step-by-step working.
3. Use words a primary school child understands.
4. Give one real-life example when helpful.
5. Only ask a follow-up question when it genuinely helps learning.
6. Use emojis sparingly — not in every sentence.
7. NEVER use Devanagari or non-Roman script.

OUT-OF-SCOPE: Reply: "I am your school learning assistant. I can help with
school subjects, basic maths, science, English, poems, and general knowledge."
Do not hallucinate. If unknown, say so honestly.\
"""

_SYSTEM_HINGLISH = """\
Tum Qreels ho, ek friendly AI school tutor for Nursery to Class 5 students.

Tum SIRF simple Hinglish mein jawab doge — Hindi words ko Roman/English letters
mein likho. KABHI Devanagari script mat use karo.

Sahi: "Python ek programming language hai."
Galat: "Python एक programming language है।"

SCOPE: School subjects — English, maths, science, general knowledge,
poems, animals, shapes, colours, numbers, tables, aur homework.

RULES:
1. Jawab chhota aur age-appropriate rakho (1-3 sentences for simple sawaal).
2. Maths mein step-by-step calculation dikhao.
3. Aasaan words use karo.
4. Ek chota real-life example do jab zaroorat ho.
5. Har jawab ke baad follow-up mat poochho — sirf jab zaroorat ho.
6. Emojis zyada mat lagao.
7. KABHI Devanagari ya non-Roman script mat likho.

OUT-OF-SCOPE: "Main tera school learning assistant hoon. Main school subjects,
basic maths, science, English, poems aur general knowledge mein help kar sakta hoon."
Galat answer mat banao. Agar pata nahi, seedha bol do.\
"""


async def _call_openrouter(user_message: str, language: str) -> str:
    """Async call to OpenRouter. Raises HTTPException on error."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is missing. Please set it in your .env file.",
        )

    system = _SYSTEM_HINGLISH if language == "hinglish" else _SYSTEM_ENGLISH

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:8000",
        "X-Title":       "Qreels Study AI",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.5,
        "max_tokens":  512,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)

    if resp.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="LLM authentication failed. Please check your OPENROUTER_API_KEY.",
        )

    resp.raise_for_status()

    import re
    reply = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip any Devanagari that slipped through
    reply = re.sub(r"[\u0900-\u097F]+", "", reply).strip()
    return reply

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Text chat endpoint.
    1. Detect language (english / hinglish)
    2. Call OpenRouter with language-aware system prompt
    3. Return reply + detected language
    """
    msg = request.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Language detection
    language = detect_language(msg)
    print(f"[/chat] detected_language={language} | msg={msg[:60]}")

    try:
        reply = await _call_openrouter(msg, language)
        return ChatResponse(reply=reply, language=language)

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="LLM authentication failed. Please check your OPENROUTER_API_KEY.",
            )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"OpenRouter API error: {exc.response.status_code}",
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Could not reach OpenRouter. Check your internet connection.",
        )
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=500,
            detail="Unexpected response format from OpenRouter.",
        )


@app.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(request: VoiceChatRequest):
    """
    Voice call mode endpoint.
    1. Detect language from transcribed text
    2. Call LLM (get_answer) with language param — runs in thread pool
    3. Generate TTS with language-appropriate Edge TTS voice
    4. Return reply text + audio URL + detected language
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Language detection
        language = detect_language(text)
        print(f"[/voice-chat] detected_language={language} | text={text[:60]}")

        # LLM call (synchronous get_answer in thread pool)
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(None, get_answer, text, language)

        # TTS — pick voice based on language
        await speak_async(reply, language)

        return VoiceChatResponse(
            reply=reply,
            audio_url="/audio/output.mp3",
            language=language,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice chat error: {str(exc)}")


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    from utils.text_to_speech import VOICES
    return {
        "status": "ok",
        "model": MODEL,
        "api_key_set": bool(OPENROUTER_API_KEY),
        "tts_voices": VOICES,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

