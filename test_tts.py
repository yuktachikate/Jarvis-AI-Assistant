import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables
os.environ["WEB_UI_MODE"] = "true"

# Import after setting environment variables
print("Importing jarvis modules...")
import jarvis
print("Modules imported")

print("Getting TTS queue and worker...")
from jarvis import tts_queue, tts_worker
print("TTS components retrieved")

import threading
import time

# Start TTS worker thread
print("Starting TTS worker thread...")
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()
print("Thread started")

# Queue up a test message
print("Sending test message to TTS queue...")
tts_queue.put("Hello! This is a test message from Jarvis. Can you hear me?")
print("Message sent")

# Keep the main thread alive for a bit
for i in range(10):
    print(f"Waiting... {i+1}/10")
    time.sleep(1)
print("Test complete.")
