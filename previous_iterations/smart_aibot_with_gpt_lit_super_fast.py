import warnings
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead",
    category=UserWarning
)

import os
import threading
import time
import logging
import random
import wave
import numpy as np
import sounddevice as sd
import queue
import whisper
import webrtcvad
from openai import OpenAI
from TTS.api import TTS

# ------------------------
# Configuration
# ------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

MEETING_ID     = "95936178307"
MEETING_PWD    = "mgSg64"
AUDIO_DEVICE   = "BlackHole 16ch"
SAMPLE_RATE    = 16000
TRANSCRIPT_LOG = "meeting_transcript.txt"

# Whisper model for speed
WHISPER_MODEL  = "small.en"
FRAME_MS       = 20
FRAME_BYTES    = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
SILENCE_FRAMES = int(0.5 * 1000 / FRAME_MS)

TTS_MODEL      = "tts_models/en/vctk/vits"
MALE_SPEAKER   = "p226"

PLACEHOLDER_PHRASE = "Hang on a minute? Asking my intelligence."

ZOOM_URL       = f"zoommtg://zoom.us/join?confno={MEETING_ID}&pwd={MEETING_PWD}"

INTRO_GREETING = (
    "Hello everyone, I’m Jarvis, your personal meeting assistant. "
    "I will be transcribing this session and standing by for any call outs. "
    "You can talk with me by saying 'Hey, Jarvis', 'Hello, Jarvis', or 'What's up, Jarvis'. "
    "Or if you want to be more casual, 'Yo, Jarvis' or 'Sup, Jarvis'. "
    "Nice to meet you all. Let's get started!"
)

GREETINGS = [
    "Hello—what can I do for you?",
    "I’m here, how may I assist?",
    "Yes? Tell me what you need.",
    "At your service—what’s on your mind?",
    "I’m listening—how can I help?",
    "You called? How can I be of assistance?",
    "Ready and waiting—what do you need?",
    "How can I assist you today?"
]

CALL_OUTS = [
    "hey jarvis", "hello jarvis", "hi jarvis",
    "what's up jarvis", "what's up, jarvis",
    "yo jarvis", "sup jarvis"
]

# ------------------------
logging.basicConfig(
    filename=TRANSCRIPT_LOG,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

audio_queue     = queue.Queue()
tts_queue       = queue.Queue()
playback_queue  = queue.Queue()

print("[INFO] Loading Whisper model…")
whisper_model = whisper.load_model(WHISPER_MODEL)

print("[INFO] Loading Coqui TTS model…")
tts_client = TTS(model_name=TTS_MODEL, progress_bar=False, gpu=False)

# sample rate from TTS synthesizer
TTS_SAMPLE_RATE = tts_client.synthesizer.output_sample_rate

# smoothing to avoid crackle
def smooth(wav: np.ndarray, sr: int) -> np.ndarray:
    fade_len = int(0.01 * sr)  # 10 ms
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = fade_in[::-1]
    wav[:fade_len] *= fade_in
    wav[-fade_len:] *= fade_out
    return wav

# pre-cache placeholder
_placeholder_wav = smooth(
    tts_client.tts(PLACEHOLDER_PHRASE, speaker=MALE_SPEAKER),
    TTS_SAMPLE_RATE
)

print("[INFO] Initializing VAD…")
vad = webrtcvad.Vad(2)

def tts_synthesis_worker():
    """Synthesize text into wav and enqueue for playback."""
    while True:
        phrase = tts_queue.get()
        if phrase is None:
            playback_queue.put(None)
            break
        if phrase == PLACEHOLDER_PHRASE:
            wav = _placeholder_wav
        else:
            wav = tts_client.tts(phrase, speaker=MALE_SPEAKER)
            wav = smooth(wav, TTS_SAMPLE_RATE)
        playback_queue.put(wav)

def tts_playback_worker():
    """Play wavs from synthesis pipeline."""
    while True:
        wav = playback_queue.get()
        if wav is None:
            break
        sd.play(wav, TTS_SAMPLE_RATE)
        sd.wait()

def enqueue_intro():
    tts_queue.put(INTRO_GREETING)

def enqueue_random():
    tts_queue.put(random.choice(GREETINGS))

def ask_gpt(prompt: str) -> None:
    """Stream GPT via gpt-4o-mini, push sentences immediately, then ask follow-up."""
    tts_queue.put(PLACEHOLDER_PHRASE)
    logger.info(f"Calling GPT with prompt: {prompt}")
    print(f"[DEBUG] ask_gpt() called with prompt: {prompt!r}")

    buffer = ""
    sentence_enders = {".", "?", "!"}

    try:
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            max_tokens=150,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Jarvis, a helpful meeting assistant. "
                        "Reply in plain text only, limit to 4–5 sentences, "
                        "after answering ask: 'Do you have any other questions?'"
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )

        for part in stream:
            delta = getattr(part.choices[0].delta, "content", None)
            if not delta:
                continue
            buffer += delta
            # flush complete sentences
            while any(buffer.endswith(e) for e in sentence_enders):
                for e in sentence_enders:
                    if e in buffer:
                        idx = buffer.rfind(e) + 1
                        sent = buffer[:idx].strip()
                        buffer = buffer[idx:].lstrip()
                        tts_queue.put(sent)
                        logger.info(f"GPT chunk responded: {sent}")
                        break

        if buffer.strip():
            tts_queue.put(buffer.strip())
            logger.info(f"GPT final chunk: {buffer.strip()}")

    except Exception as e:
        msg = f"GPT API error: {e}"
        print(f"[ERROR] {msg}")
        logger.error(msg)
        tts_queue.put("Sorry, I encountered an error while consulting my intelligence.")

def launch_zoom():
    time.sleep(1)
    os.system(f"open '{ZOOM_URL}'")
    time.sleep(8)

def transcribe_meeting():
    speech_buf, silent_cnt, buf = bytearray(), 0, bytearray()
    while True:
        data = audio_queue.get()
        if data is None:
            break
        buf.extend(data)
        while len(buf) >= FRAME_BYTES:
            frame, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
            if vad.is_speech(frame, SAMPLE_RATE):
                speech_buf.extend(frame)
                silent_cnt = 0
            elif speech_buf:
                silent_cnt += 1
                if silent_cnt > SILENCE_FRAMES:
                    audio = np.frombuffer(speech_buf, np.int16).astype(np.float32)/32768.0
                    speech_buf, silent_cnt = bytearray(), 0

                    res = whisper_model.transcribe(
                        audio,
                        temperature=0.0,
                        beam_size=1,
                        best_of=1
                    )
                    text = " ".join(
                        seg["text"].strip()
                        for seg in res["segments"]
                        if seg.get("no_speech_prob", 1.0) < 0.5
                    ).strip()

                    if len(text) >= 2:
                        logger.info(text)
                        print(f"[TRANSCRIPT] {text}")
                        low = text.lower().replace(",", "")
                        match = next((c for c in CALL_OUTS if low.startswith(c)), None)
                        if match:
                            rest = low[len(match):].strip(" ,.?")
                            if rest:
                                ask_gpt(rest)
                            else:
                                enqueue_random()

def main():
    threading.Thread(target=tts_synthesis_worker, daemon=True).start()
    threading.Thread(target=tts_playback_worker, daemon=True).start()
    threading.Thread(target=launch_zoom, daemon=True).start()
    threading.Thread(target=transcribe_meeting, daemon=True).start()
    threading.Timer(12, enqueue_intro).start()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_BYTES,
        dtype='int16',
        channels=1,
        device=AUDIO_DEVICE,
        callback=lambda indata, *_: audio_queue.put(bytes(indata))
    ):
        print("[INFO] Running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            audio_queue.put(None)
            tts_queue.put(None)
            playback_queue.put(None)

if __name__ == "__main__":
    main()
