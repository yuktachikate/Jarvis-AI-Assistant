import threading
import time
import sounddevice as sd
import queue
import subprocess
import webbrowser
import pyttsx3
import json
import logging
from vosk import Model, KaldiRecognizer

# ------------------------
# Configuration
# ------------------------

MEETING_ID = "89828568746"       # Replace with your Zoom meeting ID
MEETING_PWD = "V4GTqx"   # Replace with your Zoom meeting password
VOSK_MODEL_PATH = "/Users/achoudhary/Downloads/vosk-model-small-en-us-0.15"  # Update to your local Vosk model path
AUDIO_DEVICE = "BlackHole 16ch"  # Name of virtual audio device (input) configured for Zoom mic
SAMPLE_RATE = 16000              # Sample rate expected by Vosk model
TRANSCRIPT_LOG = "meeting_transcript.txt"

# Zoom URL scheme for CLI join
ZOOM_URL = f"zoommtg://zoom.us/join?confno={MEETING_ID}&pwd={MEETING_PWD}"

# ------------------------
# Initialize logging
# ------------------------
logging.basicConfig(
    filename=TRANSCRIPT_LOG,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

# ------------------------
# Helper: Launch Zoom meeting
# ------------------------
def launch_zoom_meeting():
    """Open the Zoom client and join the specified meeting."""
    webbrowser.open(ZOOM_URL)
    print(f"[INFO] Launched Zoom meeting: {MEETING_ID}")
    # Wait for the meeting client to initialize
    time.sleep(8)

# ------------------------
# Helper: Play greeting via TTS
# ------------------------
def greet_participants():
    """Speak an audio greeting into the meeting via system TTS."""
    engine = pyttsx3.init()
    # On macOS, engine uses system voices
    greeting = (
        "Hello everyone. "
        "I am your automated meeting assistant. "
        "I will now transcribe this meeting."
    )
    # Optionally adjust rate/voice
    engine.setProperty('rate', 150)
    engine.say(greeting)
    engine.runAndWait()
    print("[INFO] Greeting played.")

# ------------------------
# Speech-to-Text Transcription
# ------------------------
def transcribe_meeting(q):
    """Continuously read audio buffers from queue q and transcribe."""
    model = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    while True:
        data = q.get()
        if data is None:
            # Sentinel received: end transcription
            break

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text.strip():
                logger.info(f"{text}")
                print(f"[TRANSCRIPT] {text}")
        else:
            # You can optionally handle partial results if needed
            pass

# ------------------------
# Main: Audio Stream & Control
# ------------------------
def main():
    # Launch Zoom meeting in background
    threading.Thread(target=launch_zoom_meeting, daemon=True).start()

    # Prepare audio queue and transcription thread
    audio_queue = queue.Queue()
    transcriber = threading.Thread(target=transcribe_meeting, args=(audio_queue,), daemon=True)
    transcriber.start()

    # Wait a bit and then greet
    time.sleep(12)
    threading.Thread(target=greet_participants, daemon=True).start()

    # Start capturing audio from the virtual mic device
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=8000,
                               dtype='int16',
                               channels=1,
                               device=AUDIO_DEVICE,
                               callback=lambda indata, frames, time, status: audio_queue.put(bytes(indata))):
            print("[INFO] Transcription started. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping transcription...")
    finally:
        # Send sentinel to transcription thread and wait for it to finish
        audio_queue.put(None)
        transcriber.join()
        print("[INFO] Transcript saved to", TRANSCRIPT_LOG)

if __name__ == "__main__":
    main()

# ------------------------
# Prerequisites:
# ------------------------
# 1. Install Python dependencies:
#      pip install sounddevice vosk pyttsx3
#
# 2. Install Vosk model:
#      Download and unpack the model at VOSK_MODEL_PATH.
#
# 3. Set up BlackHole (or similar) on macOS:
#    - Route system output to BlackHole.
#    - In Zoom Audio Settings, set Microphone to BlackHole.
#
# 4. Replace MEETING_ID, MEETING_PWD, and VOSK_MODEL_PATH with your values.
#
# 5. Run this script. It will auto-launch Zoom, greet participants,
#    and log the meeting transcript to meeting_transcript.txt.
