import warnings
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead",
    category=UserWarning
)

import threading
import time
import sounddevice as sd
import queue
import webbrowser
import subprocess
import tempfile
import os
import wave
import numpy as np
import logging
import random
import whisper
import webrtcvad
from TTS.api import TTS  # Coqui TTS

# ------------------------
# Configuration
# ------------------------
MEETING_ID         = "93873644314"
MEETING_PWD        = "mgSg64"
AUDIO_DEVICE       = "BlackHole 16ch"
SAMPLE_RATE        = 16000
TRANSCRIPT_LOG     = "meeting_transcript.txt"

# Use Whisper’s medium English model
WHISPER_MODEL_NAME = "medium.en"

# Coqui multi-speaker TTS model (VCTK)
TTS_MODEL_NAME     = "tts_models/en/vctk/vits"
MALE_SPEAKER       = "p226"

# Zoom deep link for joining
ZOOM_URL = f"zoommtg://zoom.us/join?confno={MEETING_ID}&pwd={MEETING_PWD}"

# First-time intro greeting
INTRO_GREETING = (
    "Hello everyone, I’m Jarvis, your personal meeting assistant. "
    "I will be transcribing this session and standing by for any call outs. "
    "You can talk with me by saying 'Hey, Jarvis', 'Hello, Jarvis', or 'What's up, Jarvis.'. "
    "Or if you want to be more casual with me, I can also listen to 'Yo, Jarvis', or 'Sup, Jarvis'. "
    "But that depends on my mood. "
    "Nice to meet you all. "
    "Let's get started!"
)

# Subsequent follow-up greetings
GREETINGS = [
    "Hello—what can I do for you?",
    "I’m here, how may I assist?",
    "Yes? Tell me what you need.",
    "At your service—what’s on your mind?",
    "I’m listening—how can I help?",
    "You called? How can I be of assistance?",
    "Ready and waiting—what do you need?",
    "How can I assist you today?",
    "What can I do for you?",
    "I’m here to help—just let me know what you need.",
    "Greetings—how may I be of service?",
    "Hi there—what can I assist with?",
    "Your wish is my command—how can I help?",
    "I’m all ears—what’s up?",
    "Listening—what would you like me to do?",
    "How can I make your life easier?",
    "What can I help you with right now?",
    "I’m on standby—how can I assist?",
    "Need something? I’m here.",
    "At your disposal—what can I do?"
]

# Phrases that trigger a response
CALL_OUTS = [
    "hello jarvis",
    "hi jarvis",
    "what's up jarvis",
    "what's up, jarvis",
    "yo jarvis",
    "sup jarvis"
]

# ------------------------
# VAD & frame settings
# ------------------------
FRAME_MS       = 30
FRAME_BYTES    = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
SILENCE_FRAMES = int(0.8 * 1000 / FRAME_MS)

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    filename=TRANSCRIPT_LOG,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

# ------------------------
# Queues
# ------------------------
audio_queue = queue.Queue()
tts_queue   = queue.Queue()

# ------------------------
# Initialize Whisper, TTS & VAD
# ------------------------
print("[INFO] Loading Whisper model (medium.en)...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
print("[INFO] Whisper model loaded.")

print("[INFO] Loading Coqui TTS model...")
tts_client = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=False)
print("[INFO] Coqui TTS model loaded.")

vad = webrtcvad.Vad(2)  # Aggressiveness: 0–3

# ------------------------
# TTS worker
# ------------------------
def tts_worker():
    while True:
        phrase = tts_queue.get()
        if phrase is None:
            break
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tts_client.tts_to_file(text=phrase, file_path=tmp_wav.name, speaker=MALE_SPEAKER)
        subprocess.run(["afplay", tmp_wav.name])
        os.unlink(tmp_wav.name)
        print(f"[INFO] Played: {phrase}")

def enqueue_intro():
    tts_queue.put(INTRO_GREETING)
    print("[INFO] Intro enqueued.")

def enqueue_random():
    phrase = random.choice(GREETINGS)
    tts_queue.put(phrase)
    print(f"[INFO] Enqueued: {phrase}")

# ------------------------
# Zoom join
# ------------------------
def launch_zoom():
    webbrowser.open(ZOOM_URL)
    print("[INFO] Joining Zoom meeting...")
    time.sleep(8)

# ------------------------
# Transcription thread with VAD-based segmentation
# ------------------------
def transcribe_meeting():
    speech_buffer = bytearray()
    silent_count  = 0
    input_buffer  = bytearray()

    while True:
        data = audio_queue.get()
        if data is None:
            break

        input_buffer.extend(data)
        while len(input_buffer) >= FRAME_BYTES:
            frame = input_buffer[:FRAME_BYTES]
            input_buffer = input_buffer[FRAME_BYTES:]

            if vad.is_speech(frame, SAMPLE_RATE):
                speech_buffer.extend(frame)
                silent_count = 0
            elif speech_buffer:
                silent_count += 1
                if silent_count > SILENCE_FRAMES:
                    # Process completed utterance
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    with wave.open(tmp.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(speech_buffer)
                    # Load audio
                    with wave.open(tmp.name, 'rb') as wf:
                        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
                    os.unlink(tmp.name)

                    # Transcribe
                    result = whisper_model.transcribe(
                        audio,
                        word_timestamps=False,
                        temperature=0.0
                    )
                    text_parts = [
                        seg["text"].strip()
                        for seg in result.get("segments", [])
                        if seg.get("no_speech_prob", 1.0) < 0.5
                    ]
                    filtered = " ".join(text_parts).strip()

                    if len(filtered) >= 2:
                        logger.info(filtered)
                        print(f"[TRANSCRIPT] {filtered}")
                        low = filtered.lower().replace(",", "")
                        if any(low.startswith(c) for c in CALL_OUTS):
                            enqueue_random()

                    speech_buffer = bytearray()
                    silent_count  = 0

# ------------------------
# Main
# ------------------------
def main():
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=launch_zoom, daemon=True).start()
    threading.Thread(target=transcribe_meeting, daemon=True).start()
    threading.Timer(12, enqueue_intro).start()

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_BYTES,
            dtype='int16',
            channels=1,
            device=AUDIO_DEVICE,
            callback=lambda indata, frames, t_info, status: audio_queue.put(bytes(indata))
        ):
            print("[INFO] Running. Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("[INFO] Stopping...")
    finally:
        audio_queue.put(None)
        tts_queue.put(None)
        time.sleep(0.5)
        print("[INFO] Transcript saved.")

if __name__ == "__main__":
    main()
