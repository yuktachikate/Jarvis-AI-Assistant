import threading
import time
import sounddevice as sd
import queue
import webbrowser
import subprocess
import json
import logging
import random
from vosk import Model, KaldiRecognizer

# ------------------------
# Configuration
# ------------------------
MEETING_ID        = "85013731945"
MEETING_PWD       = "mgSg64"
VOSK_MODEL_PATH   = "/Users/achoudhary/Downloads/vosk-model-small-en-us-0.15"
AUDIO_DEVICE      = "BlackHole 16ch"
SAMPLE_RATE       = 16000
TRANSCRIPT_LOG    = "meeting_transcript.txt"
TTS_VOICE       = "Daniel"

ZOOM_URL = f"zoommtg://zoom.us/join?confno={MEETING_ID}&pwd={MEETING_PWD}"

# First-time intro greeting
INTRO_GREETING = (
    "Hello everyone, I’m Jarvis, your personal meeting assistant. "
    "I will be transcribing this session and standing by for any callouts. "
    "Let's get started!"
)

# Subsequent random greetings
GREETINGS = [
    "Hello everyone!",
    "Hi there, I am here.",
    "Greetings, colleagues!",
    "Hey there, I'm listening.",
    "Hello! How can I assist today?"
]

# Phrases that trigger a response
ACCEPTABLE_CALLOUTS = [
    "hello jarvis",
    "hi jarvis",
    "what's up jarvis"
]

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(
    filename=TRANSCRIPT_LOG,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

# ------------------------
# Queues for audio & TTS
# ------------------------
audio_queue = queue.Queue()
tts_queue   = queue.Queue()

# ------------------------
# TTS worker thread using macOS `say`
# ------------------------
def tts_worker():
    while True:
        phrase = tts_queue.get()
        if phrase is None:
            break
        subprocess.run(["say", "-v", TTS_VOICE, "-r", "180", phrase])
        print(f"[INFO] Greeting played: {phrase}")

def enqueue_intro():
    """Enqueue the one-time introduction greeting."""
    tts_queue.put(INTRO_GREETING)
    print(f"[INFO] Intro greeting enqueued: {INTRO_GREETING}")

def enqueue_random_greeting():
    """Enqueue a random follow-up greeting."""
    phrase = random.choice(GREETINGS)
    tts_queue.put(phrase)
    print(f"[INFO] Greeting enqueued: {phrase}")

# ------------------------
# Launch Zoom
# ------------------------
def launch_zoom_meeting():
    webbrowser.open(ZOOM_URL)
    print(f"[INFO] Launched Zoom meeting: {MEETING_ID}")
    time.sleep(8)

# ------------------------
# Transcription thread
# ------------------------
def transcribe_meeting():
    model      = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    while True:
        data = audio_queue.get()
        if data is None:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text   = result.get("text", "").strip()
            if text:
                logger.info(text)
                print(f"[TRANSCRIPT] {text}")
                # Trigger on callouts
                lowered = text.lower()
                if any(lowered.startswith(c) for c in ACCEPTABLE_CALLOUTS):
                    enqueue_random_greeting()

# ------------------------
# Main
# ------------------------
def main():
    # Start the TTS worker
    threading.Thread(target=tts_worker, daemon=True).start()

    # Launch Zoom
    threading.Thread(target=launch_zoom_meeting, daemon=True).start()

    # Start transcription
    threading.Thread(target=transcribe_meeting, daemon=True).start()

    # Enqueue the one-time intro greeting after join
    threading.Timer(12, enqueue_intro).start()

    # Capture audio
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype='int16',
            channels=1,
            device=AUDIO_DEVICE,
            callback=lambda indata, frames, t, status: audio_queue.put(bytes(indata))
        ):
            print("[INFO] Transcription running. Say 'Hello Jarvis'/'Hi Jarvis'/'What's up Jarvis' to trigger a greeting. Ctrl+C to quit.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        audio_queue.put(None)
        tts_queue.put(None)
        time.sleep(0.5)
        print(f"[INFO] Transcript saved to {TRANSCRIPT_LOG}")

if __name__ == "__main__":
    main()
