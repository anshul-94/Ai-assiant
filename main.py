"""
AI Chat Assistant - FastAPI Backend
- /chat      → text chat via OpenRouter (existing)
- /voice-chat → voice call mode: LLM reply + edge-tts MP3 generation
"""

import os
import asyncio
import edge_tts
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import existing util functions
from utils.llm_response import get_answer

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Chat Assistant", version="2.0.0")

# CORS — allow browser to call API from any origin (local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the audio/ folder so the browser can fetch the generated MP3
os.makedirs("audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

# ---------------------------------------------------------------------------
# Configuration — OpenRouter
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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",
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL          = "openai/gpt-3.5-turbo"

# edge-tts settings (mirrors utils/text_to_speech.py but without afplay)
TTS_VOICE       = "en-US-AriaNeural"
TTS_OUTPUT_FILE = "audio/output.mp3"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class VoiceChatRequest(BaseModel):
    text: str

class VoiceChatResponse(BaseModel):
    reply: str
    audio_url: str

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _generate_speech(text: str) -> None:
    """
    Convert text → MP3 using edge-tts and save to audio/output.mp3.
    Does NOT play locally — the browser plays the file via /audio/output.mp3.
    """
    import re
    clean = re.sub(r"[^\w\s.,!?'-]", "", text)   # strip markdown symbols for TTS
    if len(clean.strip()) < 3:
        return
    communicate = edge_tts.Communicate(
        text=clean,
        voice=TTS_VOICE,
        rate="-10%",
        pitch="+0Hz",
    )
    await communicate.save(TTS_OUTPUT_FILE)


async def _call_openrouter(user_message: str) -> str:
    """Call OpenRouter and return the assistant reply text."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:8000",
        "X-Title":       "AI Chat Assistant",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful, knowledgeable, and friendly AI assistant. "
                    "Provide clear, concise, and accurate answers."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens":  1024,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve index.html at the root URL."""
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Text chat endpoint — forwards message to OpenRouter and returns plain text reply.
    Used by the existing text chat UI.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        reply = await _call_openrouter(request.message)
        return ChatResponse(reply=reply)

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"OpenRouter API error: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Could not reach OpenRouter: {exc}")
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Unexpected response format from OpenRouter.")


@app.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(request: VoiceChatRequest):
    """
    Voice Call Mode endpoint.

    Flow:
    1. Receive transcribed user speech as plain text.
    2. Generate an AI reply using utils/llm_response.get_answer()
       (EduBuddy GPT-4o-mini, bilingual Hinglish/English teacher persona).
    3. Convert the reply to MP3 with edge-tts → saved to audio/output.mp3.
    4. Return the reply text + the URL to stream the audio.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Step 1 — LLM: use the existing EduBuddy prompt from utils/llm_response.py
        # get_answer() is synchronous; run in thread pool to avoid blocking the event loop
        reply = await asyncio.get_event_loop().run_in_executor(
            None, get_answer, request.text
        )

        # Step 2 — TTS: generate MP3 (browser will play it, no local afplay)
        await _generate_speech(reply)

        # Return reply text + audio URL (cache-busting is handled by the frontend)
        return VoiceChatResponse(
            reply=reply,
            audio_url="/audio/output.mp3",
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice chat error: {str(exc)}")


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "tts_voice": TTS_VOICE}
