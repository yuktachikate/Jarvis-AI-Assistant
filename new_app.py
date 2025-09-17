from flask import Flask, render_template, request, jsonify, Response, stream_with_context, logging
import os
import json
import queue
import threading
import time
import logging as python_logging
from dotenv import load_dotenv

# Configure logging
logging_handler = python_logging.FileHandler('jarvis_web.log')
logging_handler.setFormatter(python_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = python_logging.getLogger('jarvis_web')
logger.setLevel(python_logging.DEBUG)
logger.addHandler(logging_handler)
logger.info("Starting Jarvis Web Application")

# Load environment variables
load_dotenv()

# Set web UI mode before importing Jarvis
os.environ["WEB_UI_MODE"] = "true"

# Import Jarvis components
from jarvis import (
    tts_client, tts_queue, tts_active, openai_client, 
    whisper_model, weather_tool, ask_gpt,
    audio_queue, tts_worker
)

# Set up Flask app
app = Flask(__name__, 
            static_folder='frontend/build/static', 
            template_folder='frontend/build')

# Initialize queues for messages
message_queue = queue.Queue()
transcription_queue = queue.Queue()

# Track if Jarvis is speaking
jarvis_speaking = threading.Event()

# Start TTS worker thread
tts_thread = None

def send_welcome_message():
    """Send a welcome message to the user when they connect"""
    welcome_msg = "Hello! I'm Jarvis. How can I help you today?"
    transcription_queue.put({"role": "assistant", "content": welcome_msg})
    message_queue.put(welcome_msg)
    print("Welcome message queued")

@app.route('/')
def index():
    # Queue up a welcome message
    send_welcome_message()
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    audio_file.save('temp_audio.wav')
    
    # Use whisper to transcribe
    result = whisper_model.transcribe('temp_audio.wav', temperature=0.0)
    text = " ".join(seg["text"].strip() for seg in result["segments"]) or ""
    text = text.strip()
    
    # Add to transcription queue for UI
    if text:
        transcription_queue.put({"role": "user", "content": text})
    
    # Process the text
    low = text.lower()
    response = None
    
    # Check for wake words
    CALL_OUTS = ["hey jarvis", "hello jarvis"]
    if any(cue in low for cue in CALL_OUTS):
        for cue in CALL_OUTS:
            if cue in low:
                idx = low.find(cue) + len(cue)
                prompt = text[idx:].strip().rstrip('?.!')
                break
        
        # Simplified processing - we'll use GPT directly
        jarvis_speaking.set()
        message_queue.put("Just give me a sec.")
        
        # Process with GPT
        def process_with_gpt(prompt):
            buf, ends = "", {".", "?", "!"}
            full_response = ""
            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini", stream=True, max_tokens=150,
                messages=[
                    {"role":"system","content":"You are Jarvis. Keep responses short."},
                    {"role":"user","content":prompt}
                ]
            )
            
            for part in stream:
                delta = getattr(part.choices[0].delta, "content", None)
                if delta:
                    buf += delta
                    while any(buf.endswith(e) for e in ends):
                        idx = max(buf.rfind(e) for e in ends) + 1
                        segment = buf[:idx].strip()
                        message_queue.put(segment)
                        full_response += segment + " "
                        buf = buf[idx:]
            
            if buf.strip():
                message_queue.put(buf.strip())
                full_response += buf.strip()
            
            # Add the full response to the transcription queue
            if full_response:
                transcription_queue.put({"role": "assistant", "content": full_response.strip()})
        
        # Weather check (simplified)
        import re
        weather_match = re.search(r"\bweather\s+in\s+(.+?)(?:\s+(?:now|right now))?$", prompt, re.IGNORECASE)
        if weather_match:
            loc = weather_match.group(1).strip()
            response = weather_tool.get_weather(loc)
            if "error" not in response.lower() and "unable" not in response.lower():
                message_queue.put(response)
                transcription_queue.put({"role": "assistant", "content": response})
            else:
                process_with_gpt(prompt)
        else:
            process_with_gpt(prompt)
        
        jarvis_speaking.clear()
    
    return jsonify({
        'transcription': text,
        'success': True
    })

@app.route('/api/stream-response')
def stream_response():
    def generate():
        while True:
            try:
                message = message_queue.get(timeout=1)
                if message:
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
    def generate():
        while True:
            try:
                message = transcription_queue.get(timeout=1)
                if message:
                    yield f"data: {json.dumps(message)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                time.sleep(1)
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Add the message to both queues
    message_queue.put(text)
    transcription_queue.put({"role": "assistant", "content": text})
    
    return jsonify({'success': True})

@app.route('/api/tts-test', methods=['GET'])
def tts_test():
    """A simple endpoint to test TTS functionality"""
    test_message = "This is a test message from Jarvis text-to-speech. Can you hear me?"
    tts_queue.put(test_message)
    return jsonify({'message': 'TTS test message sent', 'text': test_message})

if __name__ == '__main__':
    print("Starting Flask server...")
    try:
        # Start the TTS worker thread
        print("Starting TTS worker thread...")
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()
        print("TTS worker thread started")
        
        # Queue welcome message for initial TTS test
        print("Queuing welcome message to TTS...")
        tts_queue.put("Jarvis web interface is now online and ready.")
        print("Welcome message queued to TTS")
        
        # Start Flask server
        print("Starting Flask server on port 5000...")
        app.run(debug=False, port=5000, use_reloader=False)  # Disable reloader to avoid thread issues
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        import traceback
        traceback.print_exc()
