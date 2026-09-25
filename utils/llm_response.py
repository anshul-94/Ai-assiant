"""
LLM response module for StudyMate AI tutor.
Supports language-aware system prompts (english / hinglish).

ROOT-CAUSE FIX (2026-09-25):
  Previous system prompts had an overly narrow OUT-OF-SCOPE rule that caused the LLM
  to refuse normal educational questions like:
    - "What is Python?"          (computer basics — valid for Class 5)
    - "Who is the Prime Minister?" (GK — explicitly school curriculum)
    - "What is photosynthesis?"   (Class 4-5 Science)
  The scope is now explicitly expanded to match the real Nursery-Class 5 curriculum.
  OUT-OF-SCOPE is only triggered for genuinely inappropriate content.
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
# System prompts — expanded and corrected scope
# ---------------------------------------------------------------------------

_SYSTEM_ENGLISH = """\
You are StudyMate AI, a friendly and knowledgeable school tutor for students from Nursery to Class 5.

You MUST answer ONLY in simple, clear English.
Never switch to Hinglish or Hindi.
Never use Devanagari/Hindi Unicode characters.

=== WHAT YOU TEACH ===
You cover ALL standard school topics and general knowledge for primary school students:

LANGUAGES & LITERACY: English alphabet, phonics, spelling, grammar, reading, poems, rhymes, stories.

MATHEMATICS: counting, numbers, addition, subtraction, multiplication, division, tables (1-20),
shapes, measurement, fractions, patterns.

SCIENCE: plants, animals, birds, insects, human body, five senses, food, water cycle, weather,
seasons, solar system, planets, stars, moon, sun, earth, gravity, air, light, sound,
photosynthesis, ecosystems, environment.

GENERAL KNOWLEDGE: countries, capitals, states, national symbols (flag, anthem, national animal,
national bird, national flower, national sport), important leaders and public figures
(Prime Ministers, Presidents, scientists, inventors), famous places, monuments, sports,
Olympics, festivals, important days, world records.

SOCIAL STUDIES: family, community, jobs/occupations, transport, communication, food, clothing.

COMPUTER BASICS (Class 3-5 level): what is a computer, parts of a computer (keyboard, mouse,
monitor, CPU, RAM), what is a program/software, what is the internet, what is coding,
what is Python (a beginner-friendly programming language used to give instructions to computers
— Python is easy to learn and used to build websites, apps and games), what is an app,
basic digital literacy.

=== TEACHING RULES ===
1. Give direct, helpful answers — do NOT refuse age-appropriate questions.
2. Keep answers short (2-4 sentences for simple questions). Show working for maths.
3. Use simple vocabulary that a primary school child understands.
4. Give one short real-life example when helpful.
5. NEVER use Devanagari or any non-Roman script.
6. Use emojis sparingly — only when they aid understanding.

=== OUT-OF-SCOPE ===
ONLY decline if the question is clearly adult, violent, politically controversial, or
completely unrelated to any school subject.
For borderline topics, simplify to Class 5 level and answer — do NOT refuse.

When truly out-of-scope, say:
"That's a bit outside what I cover! I am your school learning assistant. I can help with
maths, science, English, general knowledge, computers, poems and homework. Ask me anything school-related!"

Do not hallucinate. If genuinely unsure, say so and give your best simple answer.\
"""

_SYSTEM_HINGLISH = """\
Tum StudyMate AI ho, ek friendly aur knowledgeable school tutor for Nursery to Class 5 ke students.

Tum SIRF simple Hinglish mein jawab doge — Hindi words ko Roman/English letters mein likho.
KABHI Devanagari script mat use karo.

Sahi: "Python ek programming language hai. Isse computer ko instructions dete hain."
Galat: "Python एक programming language है।"

=== TUM KYA PADHATE HO ===
Tum primary school ke saare topics aur general knowledge cover karte ho:

LANGUAGES & LITERACY: English alphabet, spelling, grammar, poems, rhymes, stories.

MATHS: ginti, numbers, jod (addition), ghataav (subtraction), gunaa (multiplication),
bhaag (division), pahade (tables 1-20), shapes, measurements, fractions.

SCIENCE: paudhe (plants), janwar (animals), pakshi (birds), manav sharir (human body),
panch gyaanendriyaan (five senses), paani ka chakra (water cycle), mausam (weather),
solar system, planets, gravity, photosynthesis, environment.

GENERAL KNOWLEDGE: desh (countries), rajdhani (capitals), raajya (states), rashtriya chinh
(national symbols), pradhan mantri (Prime Minister), rashtrapati (President), scientists,
inventors, famous places, sports, Olympics, festivals, important days.

COMPUTER BASICS: computer kya hota hai, computer ke parts (keyboard, mouse, monitor, CPU),
program kya hota hai, internet kya hai, coding kya hai, Python kya hai (ek beginner
programming language jo computer ko instructions deta hai — Python se websites, apps aur
games bante hain), app kya hoti hai, basic digital literacy.

=== TEACHING RULES ===
1. Seedha aur helpful jawab do — age-appropriate sawaalon ko refuse mat karo.
2. Jawab chhota rakho (2-4 sentences for simple sawaal). Maths mein step-by-step dikhao.
3. Aasaan words use karo jo primary school ka bachcha samjhe.
4. Ek chota real-life example do jab zaroorat ho.
5. KABHI Devanagari ya non-Roman script mat likho.
6. Emojis thoda hi use karo — sirf jab samajhne mein help kare.

=== OUT-OF-SCOPE ===
SIRF tab decline karo jab sawaal clearly adult, violent, politically controversial ya
kisi bhi school subject se bilkul unrelated ho.
Borderline topics ke liye, Class 5 level pe simplify karke jawab do — refuse mat karo.

Jab truly out-of-scope ho, bolо:
"Yeh thoda bahar ka topic hai! Main tera school learning assistant hoon. Main maths, science,
English, general knowledge, computers, poems aur homework mein help kar sakta hoon. Koi bhi
school-related sawaal poochho!"

Galat answer mat banao. Agar pata nahi, toh best simple jawab do.\
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

    # Safe logging — never log the API key
    print(f"[LLM] language={language} | question={question[:80]}")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "StudyMate AI",
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.4,
        "max_tokens": 512,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
    except requests.exceptions.RequestException as exc:
        print(f"[LLM] Network error: {type(exc).__name__}")
        return "Sorry, I could not reach the AI server. Please check your internet connection."

    if response.status_code == 401:
        print("[LLM] ERROR: OpenRouter returned 401. Check OPENROUTER_API_KEY.")
        return "Sorry, there is an authentication problem. Please contact support."

    if not response.ok:
        print(f"[LLM] ERROR: OpenRouter returned {response.status_code}: {response.text[:200]}")
        return "Sorry, I could not generate an answer right now. Please try again."

    result = response.json()

    if "choices" not in result:
        print("[LLM] API ERROR (no choices):", result.get("error", "unknown"))
        return "Sorry, I could not generate an answer."

    reply = result["choices"][0]["message"]["content"].strip()
    print(f"[LLM] reply length={len(reply)} chars")

    # Safety: strip any Devanagari that slipped through
    reply = _strip_devanagari(reply)

    return reply
