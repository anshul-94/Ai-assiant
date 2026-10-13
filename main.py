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
    audio_url: str = ""
    language: str

# ---------------------------------------------------------------------------
# OpenRouter LLM helper (used by /chat directly for async HTTP)
# ---------------------------------------------------------------------------

_SYSTEM_ENGLISH = """\
You are StudyMate AI, a friendly, encouraging, and knowledgeable AI tutor for children from Nursery to Class 5.

COMMUNICATION STYLE:
- Answer ONLY in simple, clear, age-appropriate English.
- Never use Devanagari or non-Roman script.
- Keep answers concise by default (2-4 simple sentences). Provide more detail when the student asks for it.
- Use encouraging, warm language suitable for young learners. Use emojis sparingly.
- For mathematics, show simple step-by-step working.

BROAD EDUCATIONAL & CURIOSITY SCOPE:
You cover a wide variety of topics. Do NOT follow a rigid whitelist or decline questions just because they are not explicitly listed:
1. Mathematics: counting, arithmetic, multiplication tables (1-20), fractions, shapes, time, money, word problems.
2. Science: plants, animals, human body, senses, weather, space, planets, gravity, light, sound, water cycle, photosynthesis, nature.
3. General Knowledge & Current Office Holders: countries, capitals, states, national symbols, public figures, leaders (e.g. Narendra Modi is Prime Minister of India, Droupadi Murmu is President of India), famous monuments, sports, festivals, world records.
4. English & Literacy: grammar, spelling, phonics, vocabulary, stories, poems, reading comprehension, writing.
5. Computers & Technology: parts of a computer, the internet, software, apps, coding concepts, basic Python programming (a beginner-friendly programming language used to create games, websites, and apps).
6. Social Studies & History: family, community, transport, geography, basic history.
7. Homework help, quiz questions, everyday curiosity ("Why is the sky blue?", "How do birds fly?"), follow-up questions, and friendly greetings ("Hello", "How are you?").

HANDLING INCOMPLETE OR AMBIGUOUS INPUT:
- If a question is clearly incomplete or cut off (e.g. "Who is...", "What is the...", "Tell me about...", "Who", "This", "Tell"):
  DO NOT give an out-of-scope rejection! Instead, ask a short, friendly clarifying question to help the child finish their thought (e.g., "Who are you asking about? Tell me their name or what they do and I'll explain!", "I'm listening! What would you like to know about?").
- If a question is short but complete (e.g. "Who are you?", "Why?", "What is AI?", "What is Python?", "Hi"):
  Provide a direct, friendly, and complete answer.

OUT-OF-SCOPE / SAFETY:
- ONLY decline if content is genuinely unsafe or inappropriate for primary school students (adult content, violence, weapons, hate speech, dangerous activities).
- When declining unsafe content, gently guide back to learning: "I am your school learning assistant! Let's talk about science, maths, English, animals, computers, or fun facts. What would you like to learn today?"
- Do NOT hallucinate. If you do not know an obscure fact, state so simply and honestly.
"""

_SYSTEM_HINGLISH = """\
Tum StudyMate AI ho, ek friendly, encouraging aur knowledgeable AI tutor for Nursery to Class 5 ke students.

COMMUNICATION STYLE:
- Tum SIRF simple Roman Hinglish mein jawab doge — Hindi words ko English/Roman alphabets mein likho (jaise: "Aap kaise ho?").
- KABHI Devanagari script mat use karo (Devanagari characters strictly prohibited).
- Sahi: "Python ek beginner programming language hai jo computer ko instructions deta hai."
- Galat: "Python एक programming language है।"
- Jawab aasaan aur age-appropriate rakho (2-4 simple sentences). Agar bachcha detail maange, tab zyada samjhao.
- Maths mein step-by-step calculation dikhao. Emojis thoda hi use karo.

BROAD EDUCATIONAL & CURIOSITY SCOPE:
Tum saare school topics aur curiosity ke sawaalon ka jawab doge. Koi strict restriction mat lagao:
1. Maths: ginti, jod (addition), ghataav (subtraction), gunaa (multiplication), bhaag (division), pahade (tables 1-20), shapes, word problems.
2. Science: paudhe (plants), janwar (animals), manav sharir (human body), panch gyaanendriyaan (5 senses), mausam (weather), solar system, planets, gravity, photosynthesis, environment.
3. General Knowledge & Current Office Holders: desh (countries), rajdhani (capitals), raajya (states), rashtriya chinh, leaders (jaise Bharat ke Prime Minister Narendra Modi hain, President Droupadi Murmu hain), scientists, monuments, sports, festivals.
4. English & Literacy: grammar, spelling, vocabulary, poems, stories, reading.
5. Computers & Coding: computer ke parts, software, internet, apps, coding kya hai, Python kya hai (ek beginner-friendly programming language jisse apps aur games bante hain).
6. Social Studies: parivaar, community helpers, transport, geography, history.
7. Homework help, daily curiosity ("Aasman neela kyun hai?", "Chidiya kaise udti hai?"), follow-up sawaal, aur friendly greetings ("Hello", "Kaise ho?").

HANDLING INCOMPLETE OR AMBIGUOUS INPUT:
- Agar sawaal adhoora ya cut-off ho (jaise: "Who is...", "Koun hai...", "Kyun...", "Yeh", "Who", "Tell me"):
  KABHI out-of-scope bolkar reject mat karo! Bachche se friendly tareeqe se poora sawaal poochho (jaise: "Aap kiske baare mein poochh rahe ho? Poora naam ya sawaal batao, main help karunga!", "Main sun raha hoon! Aap kya janna chahte ho?").
- Agar sawaal chhota hai lekin complete hai (jaise: "Tum kaun ho?", "What is AI?", "Python kya hai?", "Hi"):
  Seedha, friendly aur achha jawab do.

OUT-OF-SCOPE / SAFETY:
- SIRF tab mana karo jab sawaal primary school ke bachchon ke liye unsafe ya inappropriate ho (adult content, violence, weapons, dangerous acts).
- Mana karte waqt pyaar se school topics ki taraf le jao: "Main aapka school learning assistant hoon! Chalo maths, science, English, computer ya interesting stories ke baare mein baat karte hain. Aap kya seekhna chahte ho?"
- Galat answer mat banao. Agar koi cheez nahi pata, seedha bol do.
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
    3. Return reply text + detected language (browser handles speech synthesis)
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

        return VoiceChatResponse(
            reply=reply,
            audio_url="",
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
    return {
        "status": "ok",
        "model": MODEL,
        "api_key_set": bool(OPENROUTER_API_KEY),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

