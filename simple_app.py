from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import json
import queue
import threading
import time
import logging

# Set up logging
logging.basicConfig(
    filename='jarvis_web.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set web UI mode before importing Jarvis
os.environ["WEB_UI_MODE"] = "true"

# Import TTS components
from TTS.api import TTS
import sounddevice as sd
import numpy as np

# Initialize TTS
tts_client = None
TTS_SAMPLE_RATE = None

try:
    tts_client = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
    TTS_SAMPLE_RATE = tts_client.synthesizer.output_sample_rate
    logging.info(f"TTS initialized with sample rate {TTS_SAMPLE_RATE}")
except Exception as e:
    logging.error(f"Error initializing TTS: {e}")
    import traceback
    traceback.print_exc()

# Set up queues and flags
tts_queue = queue.Queue()
message_queue = queue.Queue()
transcription_queue = queue.Queue()
tts_active = threading.Event()

# Set up Flask app
app = Flask(__name__, 
            static_folder='frontend/build/static', 
            template_folder='frontend/build')

# TTS worker function
def tts_worker():
    logging.info("TTS worker thread started")
    try:
        with sd.OutputStream(
            samplerate=TTS_SAMPLE_RATE,
            channels=1,
            dtype='float32'
        ) as stream:
            logging.info("Audio output stream opened")
            while True:
                try:
                    phrase = tts_queue.get(timeout=10)
                    if phrase is None:
                        break
                        
                    logging.info(f"TTS processing phrase: '{phrase}'")
                    tts_active.set()
                    
                    # Generate speech
                    wav = tts_client.tts(phrase, speaker="p226")
                    logging.info(f"TTS generated {len(wav)} samples")
                    
                    # Play through audio device
                    for i in range(0, len(wav), 1024):
                        if not tts_active.is_set(): 
                            break
                        stream.write(np.asarray(wav[i:i+1024], dtype=np.float32))
                    
                    logging.info("TTS playback completed")
                    tts_active.clear()
                except queue.Empty:
                    logging.debug("TTS queue empty, waiting...")
                except Exception as e:
                    logging.error(f"Error in TTS worker: {e}")
                    import traceback
                    traceback.print_exc()
    except Exception as e:
        logging.error(f"Error setting up audio stream: {e}")
        import traceback
        traceback.print_exc()

# Routes
@app.route('/')
def index():
    logging.info("Index page requested")
    # Queue up a welcome message
    try:
        message = "Hello! I'm Jarvis. How can I help you today?"
        transcription_queue.put({"role": "assistant", "content": message})
        message_queue.put(message)
        tts_queue.put(message)
        logging.info("Welcome message queued")
    except Exception as e:
        logging.error(f"Error sending welcome message: {e}")
    return render_template('index.html')

@app.route('/api/tts-test')
def tts_test():
    """A simple endpoint to test TTS functionality"""
    logging.info("TTS test endpoint called")
    try:
        test_message = "This is a test message from Jarvis text-to-speech. Can you hear me?"
        tts_queue.put(test_message)
        logging.info(f"TTS test message queued: '{test_message}'")
        return jsonify({'message': 'TTS test message sent', 'text': test_message})
    except Exception as e:
        logging.error(f"Error in TTS test endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak():
    logging.info("Speak endpoint called")
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            logging.warning("No text provided to speak endpoint")
            return jsonify({'error': 'No text provided'}), 400
        
        # Add the message to both queues
        message_queue.put(text)
        transcription_queue.put({"role": "assistant", "content": text})
        tts_queue.put(text)
        
        logging.info(f"Message queued for speaking: '{text}'")
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error in speak endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream-response')
def stream_response():
    logging.info("Stream response endpoint connected")
    def generate():
        while True:
            try:
                message = message_queue.get(timeout=1)
                if message:
                    logging.debug(f"Sending message to client: '{message}'")
                    yield f"data: {json.dumps({'message': message, 'speaking': True})}\n\n"
                    # Simulate the time it takes to speak
                    time.sleep(len(message) * 0.05)  # Rough estimate
                    yield f"data: {json.dumps({'speaking': False})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                time.sleep(1)
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/stream-transcription')
def stream_transcription():
    logging.info("Stream transcription endpoint connected")
    def generate():
        while True:
            try:
                message = transcription_queue.get(timeout=1)
                if message:
                    logging.debug(f"Sending transcription to client: {message}")
                    yield f"data: {json.dumps(message)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                time.sleep(1)
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == '__main__':
    logging.info("Starting Jarvis Web Application")
    
    # Start TTS worker thread
    tts_thread = None
    try:
        logging.info("Starting TTS worker thread")
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()
        logging.info("TTS worker thread started")
        
        # Send a startup message
        startup_msg = "Jarvis web interface is now online and ready."
        logging.info(f"Sending startup message: '{startup_msg}'")
        tts_queue.put(startup_msg)
        
        # Start Flask server
        logging.info("Starting Flask server on port 5000")
        app.run(debug=False, port=5000, use_reloader=False)
    except Exception as e:
        logging.error(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
