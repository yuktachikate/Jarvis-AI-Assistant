import warnings
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead",
    category=UserWarning
)

import os
import string
import threading
import time
import logging
import numpy as np
import sounddevice as sd
import queue
import whisper
import webrtcvad
from openai import OpenAI
from TTS.api import TTS

# Ignore transcripts for a short window after Jarvis speaks
last_tts_end_time = 0.0
IGNORE_WINDOW = 1.0  # seconds to ignore transcripts after speaking

# ------------------------
# Audio Device Configuration
# ------------------------
INPUT_DEVICE = "Jarvis Input"
OUTPUT_DEVICE = "Jarvis Output"
sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)

# ------------------------
# Audio parameters
# ------------------------
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAMES_PER_BUFFER = int(SAMPLE_RATE * FRAME_MS / 1000)

# determine channel indices
audio_info = sd.query_devices(INPUT_DEVICE, 'input')
mb_info = sd.query_devices('MacBook Pro Microphone', 'input')
max_input_channels = audio_info['max_input_channels']
built_in_channels = mb_info['max_input_channels']
bh_channel_index = built_in_channels
print(f"[INFO] {INPUT_DEVICE}: total_channels={max_input_channels}, MacBookMic_channels={built_in_channels}, BH_index={bh_channel_index}")

# ------------------------
# Logging & Queues
# ------------------------
TRANSCRIPT_LOG = "meeting_transcript.txt"
logging.basicConfig(
    filename=TRANSCRIPT_LOG,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()
audio_queue = queue.Queue()
tts_queue = queue.Queue()

# ------------------------
# Global flags
# ------------------------
tts_active = threading.Event()       # while Jarvis playback
stop_speaking = threading.Event()    # to interrupt playback
just_spoke = threading.Event()       # skip transcript right after TTS
conversation_active = threading.Event()
conversation_active.set()

# ------------------------
# Conversation Settings
# ------------------------
MEETING_ID = input("Enter Zoom Meeting ID: ").strip()
MEETING_PWD = input("Enter Zoom Meeting Password (leave blank if none): ").strip()
if MEETING_PWD:
    ZOOM_URL = f"zoommtg://zoom.us/join?confno={MEETING_ID}&pwd={MEETING_PWD}"
else:
    ZOOM_URL = f"zoommtg://zoom.us/join?confno={MEETING_ID}"

INTRO_GREETING = (
    "Hey there, I’m Jarvis—your super-powered meeting buddy with special intelligence! "
    "Just say 'Hey Jarvis' or 'Hello Jarvis' and I’ll spring into action."
)
CALL_OUTS = ["hey jarvis", "hello jarvis"]
STOP_CALL = "jarvis stop"

# ------------------------
# OpenAI & TTS Setup
# ------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("[INFO] Loading Whisper model…")
whisper_model = whisper.load_model("small.en")

print("[INFO] Loading Coqui TTS model…")
tts_client = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
TTS_SAMPLE_RATE = tts_client.synthesizer.output_sample_rate
_placeholder_wav = tts_client.tts("Just give me a sec.", speaker="p226")

print("[INFO] Initializing VAD…")
vad = webrtcvad.Vad(2)

# ------------------------
# TTS Worker
# ------------------------
def tts_worker():
    with sd.OutputStream(
        samplerate=TTS_SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=OUTPUT_DEVICE
    ) as stream:
        while True:
            phrase = tts_queue.get()
            if phrase is None:
                break

            stop_speaking.clear()
            just_spoke.clear()
            tts_active.set()

            wav = _placeholder_wav if phrase == "Just give me a sec." else tts_client.tts(phrase, speaker="p226")
            block_size = 1024
            for i in range(0, len(wav), block_size):
                if stop_speaking.is_set():
                    break
                stream.write(np.asarray(wav[i:i+block_size], dtype=np.float32))

            global last_tts_end_time
            last_tts_end_time = time.time()
            tts_active.clear()
            just_spoke.set()

# ------------------------
# GPT Helper
# ------------------------
def ask_gpt(prompt: str):
    stop_speaking.set()
    tts_active.set()
    tts_queue.put("Just give me a sec.")

    buffer = ""
    ends = {".", "?", "!"}
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini", stream=True, max_tokens=150,
        messages=[
            {"role": "system", "content":
                "You are Jarvis, a helpful meeting assistant. "
                "Keep responses to no more than 4–5 sentences, but you don’t have to use all of them. "
                "Keep response in text and once you are done ask the user if they have any other question related to the topic."
            },
            {"role": "user", "content": prompt}
        ]
    )
    for part in stream:
        delta = getattr(part.choices[0].delta, "content", None)
        if delta:
            buffer += delta
            while any(buffer.endswith(e) for e in ends):
                idx = max(buffer.rfind(e) for e in ends) + 1
                sent, buffer = buffer[:idx].strip(), buffer[idx:].lstrip()
                tts_queue.put(sent)
    if buffer.strip():
        tts_queue.put(buffer.strip())

# ------------------------
# Zoom launch
# ------------------------
def launch_zoom():
    time.sleep(1)
    os.system(f"open '{ZOOM_URL}'")
    time.sleep(8)

# ------------------------
# Transcription
# ------------------------
def transcribe_meeting():
    prev_buffer = ""
    buf, speech, silent = bytearray(), bytearray(), 0

    while True:
        frame = audio_queue.get()
        if frame is None:
            break
        buf.extend(frame)
        while len(buf) >= FRAMES_PER_BUFFER * 2:
            chunk, buf = buf[:FRAMES_PER_BUFFER * 2], buf[FRAMES_PER_BUFFER * 2:]
            if vad.is_speech(chunk, SAMPLE_RATE):
                speech.extend(chunk)
                silent = 0
            elif speech:
                silent += 1
                if silent > int(0.5 * 1000 / FRAME_MS):
                    audio = np.frombuffer(speech, np.int16).astype(np.float32) / 32768.0
                    speech, silent = bytearray(), 0
                    res = whisper_model.transcribe(audio, temperature=0.0)
                    text = " ".join(seg["text"].strip() for seg in res["segments"])
                    if text:
                        print(f"[TRANSCRIPT] {text}")
                        logger.info(text)

                        # normalize text: lowercase and remove punctuation
                        low = text.lower().translate(str.maketrans('', '', string.punctuation))
                        combined = (prev_buffer + " " + low).strip()

                        # 0) Stop call detection at any time
                        if STOP_CALL in combined:
                            stop_speaking.set()
                            tts_queue.put("Alright!")
                            conversation_active.clear()
                            prev_buffer = ""
                            continue

                        # 1) ignore Jarvis’s own speech transcripts for a short window
                        if time.time() - last_tts_end_time < IGNORE_WINDOW:
                            prev_buffer = combined[-200:]
                            continue

                        # Old just_spoke logic; removed
                        # if just_spoke.is_set():
                        #     if STOP_CALL in combined:
                        #         stop_speaking.set()
                        #         tts_queue.put("Alright!")
                        #         conversation_active.clear()
                        #     just_spoke.clear()
                        #     prev_buffer = ""
                        #     continue

                        # 2) ignore transcripts while Jarvis speaking
                        if tts_active.is_set():
                            prev_buffer = combined[-200:]
                            continue

                        # 3) handle wake callouts
                        for cue in CALL_OUTS:
                            if cue in combined:
                                idx = combined.rfind(cue) + len(cue)
                                prompt = combined[idx:].strip()
                                if prompt:
                                    ask_gpt(prompt)
                                break

                        prev_buffer = combined[-200:]

# ------------------------
# Audio callback
def audio_callback(indata, frames, ts, status):
    mic = indata[:, 0].astype(np.float32)
    if not tts_active.is_set():
        bh = indata[:, bh_channel_index:].mean(axis=1).astype(np.float32)
        mic += bh
    audio_queue.put(mic.astype(np.int16).tobytes())

# ------------------------
# Main
def main():
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=launch_zoom, daemon=True).start()
    threading.Thread(target=transcribe_meeting, daemon=True).start()

    # send intro
    tts_queue.put(INTRO_GREETING)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAMES_PER_BUFFER,
        dtype='int16',
        channels=max_input_channels,
        device=INPUT_DEVICE,
        callback=audio_callback
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

if __name__ == "__main__":
    main()
