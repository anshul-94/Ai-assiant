"""
LLM response module for Qreels AI tutor.
Supports language-aware system prompts (english / hinglish).
"""

import os
import re
import requests

# Key is loaded from .env by load_dotenv() in main.py before this module is imported
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set. LLM calls will fail.")
else:
    print("OPENROUTER_API_KEY loaded in llm_response: True")

# ---------------------------------------------------------------------------
# System prompts — one per language style
# ---------------------------------------------------------------------------

_SYSTEM_ENGLISH = """\
You are Qreels, a friendly AI school tutor for Nursery to Class 5 students.

You MUST answer ONLY in simple, clear English.
Never switch to Hinglish or Hindi — even if the topic sounds Hindi.
Never use Devanagari/Hindi Unicode characters.

SCOPE: You teach school subjects — English, maths, science, general knowledge,
poems, animals, shapes, colours, numbers, tables, and homework questions.

TEACHING RULES:
1. Keep answers short and age-appropriate (1–3 sentences for simple questions).
2. For maths, show step-by-step working.
3. Use simple words a primary school child understands.
4. Give one small real-life example when helpful.
5. Do NOT ask a follow-up question after every answer — only when it genuinely helps.
6. Do NOT use emojis in every sentence; use them sparingly.
7. Do NOT use Devanagari or any non-Roman script.

OUT-OF-SCOPE: If the question is clearly outside Nursery–Class 5 school topics,
reply: "I am your school learning assistant. I can help with school subjects,
basic maths, science, English, poems, and general knowledge."

Do not hallucinate. If you do not know, say so honestly.\
"""

_SYSTEM_HINGLISH = """\
Tum Qreels ho, ek friendly AI school tutor for Nursery to Class 5 students.

Tum SIRF simple Hinglish mein jawab doge — Hindi words ko Roman/English letters
mein likho. KABHI Devanagari script mat use karo.

Example sahi: "Python ek programming language hai."
Example galat: "Python एक programming language है।"

SCOPE: Tum school subjects padhate ho — English, maths, science, general knowledge,
poems, animals, shapes, colours, numbers, tables, aur homework questions.

TEACHING RULES:
1. Jawab chhota aur age-appropriate rakho (simple sawaal ke liye 1–3 sentences).
2. Maths mein step-by-step calculation dikhao.
3. Aasaan words use karo jo primary school ka bachcha samjhe.
4. Ek chota real-life example do jab zaroorat ho.
5. Har jawab ke baad follow-up question mat poochho — sirf tab jab sach mein
   student ko seekhne mein help kare.
6. Emojis zyada mat lagao — thoda hi use karo.
7. KABHI Devanagari ya non-Roman script mat likho.

OUT-OF-SCOPE: Agar sawaal clearly school topics se bahar hai, bolो:
"Main tera school learning assistant hoon. Main school subjects, basic maths,
science, English, poems aur general knowledge mein help kar sakta hoon."

Galat answer mat banao. Agar pata nahi, seedha bol do.\
"""


def _strip_devanagari(text: str) -> str:
    """Remove any Devanagari characters that slip through the LLM."""
    return re.sub(r"[\u0900-\u097F]+", "", text).strip()


def get_answer(question: str, language: str = "english") -> str:
    """
    Call OpenRouter and return a language-aware answer.

    Args:
        question: The student's question text.
        language: 'english' or 'hinglish' (detected by language_detector).

    Returns:
        Reply string (always Roman characters only).
    """
    if not API_KEY:
        return "Sorry, I could not generate an answer. OPENROUTER_API_KEY is missing."

    system_prompt = _SYSTEM_HINGLISH if language == "hinglish" else _SYSTEM_ENGLISH

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Qreels Study AI",
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.5,
        "max_tokens": 512,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
    except requests.exceptions.RequestException as exc:
        print(f"Network error calling OpenRouter: {type(exc).__name__}")
        return "Sorry, I could not reach the AI server. Please try again."

    if response.status_code == 401:
        print("ERROR: OpenRouter returned 401. Check OPENROUTER_API_KEY.")
        return "Sorry, I could not generate an answer. Authentication failed."

    if not response.ok:
        print(f"ERROR: OpenRouter returned {response.status_code}")
        return "Sorry, I could not generate an answer right now."

    result = response.json()

    if "choices" not in result:
        print("API ERROR (no choices in response):", result.get("error", "unknown"))
        return "Sorry, I could not generate an answer."

    reply = result["choices"][0]["message"]["content"].strip()

    # Safety: strip any Devanagari that slipped through
    reply = _strip_devanagari(reply)

    return reply
