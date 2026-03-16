from faster_whisper import WhisperModel

# load model
model = WhisperModel("small", compute_type="int8")

def transcribe(audio):

    segments, info = model.transcribe(
        audio,
        language="en",      
        beam_size=5
    )

    text = ""

    for segment in segments:
        text += segment.text + " "

    return text.strip()
