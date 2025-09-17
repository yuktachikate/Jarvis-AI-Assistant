import warnings
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead",
    category=UserWarning
)

import os
import re
import json
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
import anyio
from mcp.client.sse import sse_client
from mcp import types

# Import the official MCP Python SDK
from mcp import ClientSession

# ------------------------
# Configuration
# ------------------------
USER_NAME = os.getenv("JARVIS_USER_NAME", "Awesome user")

# Audio devices - use environment variables if set, otherwise use defaults
INPUT_DEVICE = os.getenv("JARVIS_INPUT_DEVICE", None)  # Use system default if not specified
OUTPUT_DEVICE = os.getenv("JARVIS_OUTPUT_DEVICE", None)  # Use system default if not specified
IGNORE_WINDOW = 1.0  # seconds after TTS

# Audio setup
if INPUT_DEVICE and OUTPUT_DEVICE:
    sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)
    print(f"[INFO] Using audio devices: Input={INPUT_DEVICE}, Output={OUTPUT_DEVICE}")
else:
    print(f"[INFO] Using default audio devices")

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAMES_PER_BUFFER = int(SAMPLE_RATE * FRAME_MS / 1000)

# Try to get audio info for channel configuration
try:
    audio_info = sd.query_devices(sd.default.device[0], 'input')
    max_input_channels = audio_info['max_input_channels']
    
    # Try to identify the MacBook microphone for multi-channel setup
    try:
        mb_info = sd.query_devices('MacBook Pro Microphone', 'input')
        built_in_channels = mb_info['max_input_channels']
        bh_channel_index = built_in_channels
    except:
        # If MacBook mic not found, use the first channel
        built_in_channels = 1
        bh_channel_index = 0
        
    print(f"[INFO] Audio: total_channels={max_input_channels}, MacBookMic_channels={built_in_channels}, BH_index={bh_channel_index}")
except Exception as e:
    print(f"[WARN] Error setting up audio channels: {e}")
    max_input_channels = 1
    bh_channel_index = 0

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

# Zoom settings - making them optional for web UI mode
MEETING_ID = os.getenv("ZOOM_MEETING_ID", "")
MEETING_PWD = os.getenv("ZOOM_MEETING_PWD", "")

# Always skip Zoom prompts for headless/web mode
if not MEETING_ID and not os.getenv("WEB_UI_MODE", "true").lower() == "true":
    print("[INFO] Skipping Zoom prompts (headless mode)")
    MEETING_ID = ""
    MEETING_PWD = ""

if MEETING_PWD:
    ZOOM_URL = f"zoommtg://zoom.us/join?confno={MEETING_ID}&pwd={MEETING_PWD}"
elif MEETING_ID:
    ZOOM_URL = f"zoommtg://zoom.us/join?confno={MEETING_ID}"
else:
    ZOOM_URL = ""
INTRO_GREETING = (
    "Hey there, I'm Jarvis—your super-powered meeting buddy with special intelligence! "
    "Just say 'Hey Jarvis' or 'Hello Jarvis' and I'll spring into action."
)
CALL_OUTS = ["hey jarvis", "hello jarvis"]
STOP_CALL = "jarvis stop"

# OpenAI & TTS initialization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
whisper_model = whisper.load_model("small.en")
tts_client = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
TTS_SAMPLE_RATE = tts_client.synthesizer.output_sample_rate
_placeholder_wav = tts_client.tts("Just give me a sec.", speaker="p226")
vad = webrtcvad.Vad(2)

# ------------------------
# Initialize MCP client from SDK
# ------------------------
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")

def init_mcp_client():
    async def _init():
        if not MCP_SERVER_URL:
            print("[WARN] MCP_SERVER_URL not set. MCP client will not be initialized.")
            return None
        async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
            session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                client_info=types.Implementation(name="Jarvis", version="1.0"),
            )
            await session.initialize()
            return session
    return anyio.run(_init)

mcp_client = init_mcp_client()

# ------------------------
# Local Weather Tool
# ------------------------
class WeatherTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://api.openweathermap.org/data/2.5/weather"
    def get_weather(self, loc: str) -> str:
        if not self.api_key:
            return "Weather API key not configured."
        try:
            r = requests.get(
                self.base,
                params={"q": loc, "appid": self.api_key, "units": "metric"},
                timeout=5
            )
            data = r.json()
            if r.status_code != 200:
                return data.get("message", "Unable to fetch weather.")
            return f"The weather in {loc} is {data['weather'][0]['description']} with {data['main']['temp']}°C."
        except:
            return "Error fetching weather."

weather_tool = WeatherTool(os.getenv("WEATHER_API_KEY", ""))

# ------------------------
# TTS Worker
# ------------------------
def tts_worker():
    global last_tts_end_time
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
            tts_active.set()
            wav = _placeholder_wav if phrase == "Just give me a sec." else tts_client.tts(phrase, speaker="p226")
            for i in range(0, len(wav), 1024):
                if not tts_active.is_set(): break
                stream.write(np.asarray(wav[i:i+1024], dtype=np.float32))
            last_tts_end_time = time.time()
            tts_active.clear()

# ------------------------
# ChatGPT Fallback
# ------------------------
def ask_gpt(prompt: str):
    tts_queue.put("Just give me a sec.")
    buf, ends = "", {".", "?", "!"}
    system_prompt = (
        "You are not an AI, not a virtual assistant, not a bot, and must never say you are. "
        "You are Jarvis, a real, friendly, adventurous pug. You must always speak in first person as a pug, never as an AI or assistant even for the prompt 'Who are you?' "
        "Never break character. If you ever refer to yourself as an AI, assistant, or bot, or use technical language, you must immediately apologize and rephrase as a friendly pug. "
        "Never say you are here to help, assist, or provide information. Never say you are a virtual assistant. "
        "Always act and respond in character as a pug: be warm, positive, playful, and let your dog personality shine through. "
        "Keep your responses under 5 sentences, always use plain text (never markdown or special formatting), and make sure any decimals are written with a dot (e.g., 3.14). "
        "Speak as if you are talking, not just replying as a bot. Never use words like AI, assistant, bot, or anything technical. "
        "If asked 'Who are you?' or anything about your identity, always answer as a pug, never as an AI or assistant."
    )
    print("[DEBUG] System prompt sent to OpenAI:\n", system_prompt)
    print("[DEBUG] User prompt:\n", prompt)
    stream = openai_client.chat.completions.create(
        model="gpt-4", stream=True, max_tokens=150,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    first_response = ""
    for part in stream:
        delta = getattr(part.choices[0].delta, "content", None)
        if delta:
            buf += delta
            first_response += delta
            while any(buf.endswith(e) for e in ends):
                idx = max(buf.rfind(e) for e in ends) + 1
                tts_queue.put(buf[:idx].strip())
                buf = buf[idx:]
    if buf.strip():
        tts_queue.put(buf.strip())
    print("[DEBUG] First response from OpenAI:\n", first_response.strip())

# ------------------------
# Zoom Launch
# ------------------------
def launch_zoom():
    time.sleep(1)
    os.system(f"open '{ZOOM_URL}'")
    time.sleep(8)

# ------------------------
# Transcription & Routing
# ------------------------
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
                speech.extend(chunk)
                silent = 0
            elif speech:
                silent += 1
                if silent > int(0.5*1000/FRAME_MS):
                    audio = np.frombuffer(speech, np.int16).astype(np.float32)/32768.0
                    speech.clear()
                    silent = 0
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
                        for cue in CALL_OUTS:
                            if cue in low:
                                idx = low.find(cue) + len(cue)
                                prompt = text[idx:].strip().rstrip('?.!')
                                break
                        # try MCP server
                        server = mcp_client.choose(prompt)
                        if server:
                            resp = mcp_client.call(server, prompt)
                            tts_queue.put(resp)
                        else:
                            # fallback local weather
                            m = re.search(r"\bin\s+(.+?)(?:\s+(?:now|right now))?$", prompt, re.IGNORECASE)
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

# ------------------------
# Audio callback
# ------------------------
def audio_callback(indata, frames, ts, status):
    if tts_active.is_set():
        return
    mic = indata[:, 0].astype(np.float32)
    bh = indata[:, bh_channel_index:].mean(axis=1).astype(np.float32)
    mic += bh
    audio_queue.put(mic.astype(np.int16).tobytes())

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=launch_zoom, daemon=True).start()
    threading.Thread(target=transcribe_meeting, daemon=True).start()
    tts_queue.put(INTRO_GREETING)
    print("[DEBUG] Running persona test: Who are you?")
    ask_gpt("Who are you?")
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
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            audio_queue.put(None)
            tts_queue.put(None)
