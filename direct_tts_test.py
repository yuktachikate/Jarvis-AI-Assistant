import os
import time
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variable for web UI mode
os.environ["WEB_UI_MODE"] = "true"

# Print diagnostics
print("Python executable:", os.sys.executable)
print("Current directory:", os.getcwd())
print("Environment variables:", os.environ.get("WEB_UI_MODE"), os.environ.get("OPENAI_API_KEY"))

try:
    # Import TTS components
    print("Importing TTS components...")
    from TTS.api import TTS
    print("TTS imported successfully")
    
    # Initialize TTS
    print("Initializing TTS client...")
    tts_client = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
    print("TTS client initialized")
    
    # Get sample rate
    TTS_SAMPLE_RATE = tts_client.synthesizer.output_sample_rate
    print(f"TTS sample rate: {TTS_SAMPLE_RATE}")
    
    # Generate test audio
    print("Generating test audio...")
    test_wav = tts_client.tts("Hello! This is a test of the text to speech system.", speaker="p226")
    print("Test audio generated successfully")
    
    # Import audio playback components
    print("Importing audio playback components...")
    import numpy as np
    import sounddevice as sd
    print("Audio playback components imported")
    
    # Set up audio output stream
    print("Setting up audio output stream...")
    stream = sd.OutputStream(
        samplerate=TTS_SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    stream.start()
    print("Audio output stream started")
    
    # Play test audio
    print("Playing test audio...")
    for i in range(0, len(test_wav), 1024):
        chunk = test_wav[i:i+1024]
        stream.write(np.asarray(chunk, dtype=np.float32))
    print("Test audio playback complete")
    
    # Clean up
    stream.stop()
    stream.close()
    print("Audio output stream closed")
    
    print("TTS test completed successfully")
except Exception as e:
    print(f"Error during TTS test: {e}")
    import traceback
    traceback.print_exc()
