import warnings
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead",
    category=UserWarning
)

import os
import re
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
import requests

# ------------------------
# Configuration
# ------------------------
USER_NAME = os.getenv("JARVIS_USER_NAME", "Awesome user")
INPUT_DEVICE = "Jarvis Input"
OUTPUT_DEVICE = "Jarvis Output"
IGNORE_WINDOW = 1.0  # seconds after TTS to ignore audio

# Audio setup
sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAMES_PER_BUFFER = int(SAMPLE_RATE * FRAME_MS / 1000)

audio_info = sd.query_devices(INPUT_DEVICE, 'input')
mb_info = sd.query_devices('MacBook Pro Microphone', 'input')
max_input_channels = audio_info['max_input_channels']
built_in_channels = mb_info['max_input_channels']
bh_channel_index = built_in_channels
print(f"[INFO] {INPUT_DEVICE}: total_channels={max_input_channels}, MacBookMic_channels={built_in_channels}, BH_index={bh_channel_index}")

# Logging & queues
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

# Flags & timing
tts_active = threading.Event()
last_tts_end_time = 0.0

# Call settings
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
CALL_OUTS = ["hey jarvis", "hello jarvis", "hey javis", "hello javis"]
STOP_CALL = "jarvis stop"

# OpenAI & TTS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

whisper_model = whisper.load_model("small.en")
tts_client = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
TTS_SAMPLE_RATE = tts_client.synthesizer.output_sample_rate
_placeholder_wav = tts_client.tts("Just give me a sec.", speaker="p226")

vad = webrtcvad.Vad(2)

# Weather tool
class WeatherTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    def get_weather(self, location: str) -> str:
        if not self.api_key:
            return "Weather API key not configured."
        try:
            r = requests.get(self.base_url,
                             params={"q": location, "appid": self.api_key, "units": "metric"},
                             timeout=5)
            data = r.json()
            if r.status_code != 200:
                return data.get("message", "Unable to fetch weather.")
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"The weather in {location} is {desc} with {temp}°Celsius."
        except Exception:
            return "Error fetching weather."

weather_tool = WeatherTool(os.getenv("WEATHER_API_KEY", ""))

# TTS thread
def tts_worker():
    global last_tts_end_time
    with sd.OutputStream(samplerate=TTS_SAMPLE_RATE,
                         channels=1, dtype='float32', device=OUTPUT_DEVICE) as stream:
        while True:
            phrase = tts_queue.get()
            if phrase is None:
                break
            tts_active.set()
            wav = _placeholder_wav if phrase == "Just give me a sec." else tts_client.tts(phrase, speaker="p226")
            for i in range(0, len(wav), 1024):
                if not tts_active.is_set():
                    break
                stream.write(np.asarray(wav[i:i+1024], dtype=np.float32))
            last_tts_end_time = time.time()
            tts_active.clear()

# GPT fallback
def ask_gpt(prompt: str):
    tts_queue.put("Just give me a sec.")
    buffer = ""
    ends = {".", "?", "!"}
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini", stream=True, max_tokens=150,
        messages=[
            {"role": "system", "content": "You are Jarvis an intelligent assistant. Keep responses under 5 sentences. At the end always ask user if they want to know more about the mentioned topic."},
            {"role": "user", "content": prompt}
        ]
    )
    for part in stream:
        delta = getattr(part.choices[0].delta, "content", None)
        if delta:
            buffer += delta
            while any(buffer.endswith(e) for e in ends):
                idx = max(buffer.rfind(e) for e in ends) + 1
                tts_queue.put(buffer[:idx].strip())
                buffer = buffer[idx:]
    if buffer.strip():
        tts_queue.put(buffer.strip())

# Launch Zoom
def launch_zoom():
    time.sleep(1)
    os.system(f"open '{ZOOM_URL}'")
    time.sleep(8)

# Transcription & handling
def transcribe_meeting():
    buf = bytearray()
    speech = bytearray()
    silent = 0
    while True:
        frame = audio_queue.get()
        if frame is None:
            break
        buf.extend(frame)
        while len(buf) >= FRAMES_PER_BUFFER * 2:
            chunk, buf = buf[:FRAMES_PER_BUFFER*2], buf[FRAMES_PER_BUFFER*2:]
            if vad.is_speech(chunk, SAMPLE_RATE):
                speech.extend(chunk); silent = 0
            elif speech:
                silent += 1
                if silent > int(0.5 * 1000 / FRAME_MS):
                    audio = np.frombuffer(speech, np.int16).astype(np.float32) / 32768.0
                    speech.clear(); silent = 0
                    res = whisper_model.transcribe(audio, temperature=0.0)
                    orig = " ".join(seg["text"].strip() for seg in res["segments"]) or ""
                    ts = time.time()
                    if tts_active.is_set() or (ts - last_tts_end_time) < IGNORE_WINDOW:
                        continue
                    text = orig.strip()
                    if not text:
                        continue
                    print(f"[TRANSCRIPT][{USER_NAME}] {text}")
                    logger.info(f"{USER_NAME}: {text}")
                    low = text.lower()
                    if STOP_CALL in low:
                        tts_queue.put("Alright!")
                        continue
                    # detect wake word
                    if any(cue in low for cue in CALL_OUTS):
                        # isolate the part after wake word
                        for cue in CALL_OUTS:
                            if cue in low:
                                idx = low.find(cue) + len(cue)
                                prompt = text[idx:].strip().rstrip('?.!')
                                break
                        # weather intent
                        m = re.search(r"\bin\s+(.+?)($|\s+(?:now|right now))", prompt, re.IGNORECASE)
                        if m:
                            loc = m.group(1).strip()
                            resp = weather_tool.get_weather(loc)
                            if re.match(r"(?i)(error|unable|weather api key|city not found)", resp):
                                ask_gpt(prompt)
                            else:
                                tts_queue.put(resp)
                        else:
                            ask_gpt(prompt)
            time.sleep(0.01)

# Audio callback
def audio_callback(indata, frames, ts, status):
    if tts_active.is_set():
        return
    mic = indata[:, 0].astype(np.float32)
    bh = indata[:, bh_channel_index:].mean(axis=1).astype(np.float32)
    mic += bh
    audio_queue.put(mic.astype(np.int16).tobytes())

# Main
if __name__ == "__main__":
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=launch_zoom, daemon=True).start()
    threading.Thread(target=transcribe_meeting, daemon=True).start()
    tts_queue.put(INTRO_GREETING)
    with sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=FRAMES_PER_BUFFER,
        dtype='int16', channels=max_input_channels,
        device=INPUT_DEVICE, callback=audio_callback
    ):
        print("[INFO] Running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            audio_queue.put(None)
            tts_queue.put(None)
