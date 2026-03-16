import asyncio
import edge_tts
import os
import re

VOICE = "en-US-AriaNeural"   # very natural female voice
OUTPUT_FILE = "audio/output.mp3"


def clean_text(text):
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text


async def speak_async(text):

    text = clean_text(text)

    if len(text.strip()) < 3:
        return

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate="-10%",   # same speed like Alexa
        pitch="+0Hz"
    )

    await communicate.save(OUTPUT_FILE)

    os.system(f"afplay {OUTPUT_FILE}")


def speak(text):
    asyncio.run(speak_async(text))
