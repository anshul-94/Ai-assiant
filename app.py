import sounddevice as sd
from scipy.io.wavfile import write

from utils.speech_to_text import transcribe
from utils.llm_response import get_answer
from utils.text_to_speech import speak

fs = 44100
duration = 6

print("🎤 AI Study Assistant started. Say 'bye' or 'exit' to stop.\n")

while True:

    try:

        print("Speak your question...")

        # record audio
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        write("audio/input.wav", fs, recording)

        question = transcribe("audio/input.wav")

        print("Student:", question)

        if question.strip() == "":
            print("No speech detected\n")
            continue

        # exit condition
        if question.lower() in ["bye", "exit", "stop", "goodbye"]:
            speak("Goodbye! Have a nice day.")
            print("Assistant: Goodbye!")
            break

        answer = get_answer(question)

        print("Assistant:", answer)

        speak(answer)

        print()

    except KeyboardInterrupt:
        print("\n🛑 Assistant stopped by user.")
        break

    except Exception as e:
        print("⚠️ Error:", e)
        continue
