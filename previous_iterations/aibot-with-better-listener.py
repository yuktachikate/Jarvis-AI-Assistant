
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

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from TTS.api import TTS  # Coqui TTS

# ------------------------
# Configuration
# ------------------------
MEETING_ID        = "93873644314"
MEETING_PWD       = "mgSg64"
AUDIO_DEVICE      = "BlackHole 16ch"
SAMPLE_RATE       = 16000
TRANSCRIPT_LOG    = "meeting_transcript.txt"

# Wav2Vec2 model for transcription
W2V_MODEL_NAME    = "facebook/wav2vec2-large-960h-lv60-self"

# Coqui multi-speaker TTS model (VCTK)
TTS_MODEL_NAME    = "tts_models/en/vctk/vits"
MALE_SPEAKER      = "p226"

# Zoom deep link for joining
ZOOM_URL = f"zoommtg://zoom.us/join?confno={MEETING_ID}&pwd={MEETING_PWD}"

# Greetings
INTRO_GREETING = (
    "Hello everyone, I’m Jarvis, your personal meeting assistant. "
    "I will be transcribing this session and standing by for any call outs. "
    "You can talk with me by saying 'Hey Jarvis', 'Hello Jarvis', or 'What's up Jarvis'. "
    "Or if you want to be more casual with me, I can also listen to 'Yo Jarvis' or 'Sup Jarvis'. "
    "But that depends on my mood. "
    "Nice to meet you all. "
    "Let's get started!"
)
GREETINGS = [
    "Hello everyone!",
    "Hi there, I am here.",
    "Greetings, colleagues!",
    "Hey there, I'm listening.",
    "Hello! How can I assist today?"
]
CALL_OUTS = ["hello jarvis", "hi jarvis", "what's up jarvis", "what's up, jarvis", "yo jarvis", "sup jarvis"]

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
# Initialize models
# ------------------------
print("[INFO] Loading Wav2Vec2 model and processor...")
processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL_NAME)
w2v_model = Wav2Vec2ForCTC.from_pretrained(W2V_MODEL_NAME)
w2v_model.eval()
if torch.cuda.is_available():
    w2v_model.to("cuda")
print("[INFO] Wav2Vec2 model loaded.")

print("[INFO] Loading Coqui TTS model...")
tts_client = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=False)
print("[INFO] Coqui TTS model loaded.")

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
# Transcription thread
# ------------------------
def transcribe_meeting():
    buffer = bytearray()
    bytes_per_frame = 2  # int16
    chunk_size = SAMPLE_RATE * 5 * bytes_per_frame  # 5-second chunks
    while True:
        data = audio_queue.get()
        if data is None:
            break
        buffer.extend(data)
        if len(buffer) >= chunk_size:
            # Save buffer to WAV
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(buffer)
            # Read WAV for processing
            with wave.open(tmp.name, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            os.unlink(tmp.name)

            # Transcription via Wav2Vec2
            input_values = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).input_values
            if torch.cuda.is_available():
                input_values = input_values.to("cuda")
            with torch.no_grad():
                logits = w2v_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = processor.decode(predicted_ids[0]).lower().strip()

            if text:
                logger.info(text)
                print(f"[TRANSCRIPT] {text}")
                if any(text.startswith(c) for c in CALL_OUTS):
                    enqueue_random()

            buffer = bytearray()

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
            blocksize=8000,
            dtype='int16',
            channels=1,
            device=AUDIO_DEVICE,
            callback=lambda indata, frames, ti, status: audio_queue.put(bytes(indata))
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
