import os
from dotenv import load_dotenv
load_dotenv()

# Set web UI mode before importing Jarvis
os.environ["WEB_UI_MODE"] = "true"

# Import Jarvis components
print("Importing Jarvis components...")
from jarvis import tts_queue, tts_worker, tts_client
import threading
import time

# Test TTS directly
def test_direct_tts():
    print("Testing TTS directly...")
    text = "This is a direct test of the TTS system."
    print(f"Converting text to speech: '{text}'")
    wav = tts_client.tts(text, speaker="p226")
    print("TTS conversion successful")
    return wav

# Start TTS worker thread
print("Starting TTS worker thread...")
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()
print("TTS worker thread started")

# Queue several test messages
messages = [
    "Hello! I am Jarvis, your AI assistant.",
    "I'm testing if my voice is working correctly.",
    "Can you hear me now? If so, the TTS system is operational."
]

# Send test messages with delays
print("Sending test messages to TTS queue...")
for i, message in enumerate(messages):
    print(f"Sending message {i+1}: '{message}'")
    tts_queue.put(message)
    # Wait for each message to be processed
    time.sleep(5)

print("All test messages sent!")
print("Waiting for final message to complete...")
time.sleep(5)
print("Test complete.")
