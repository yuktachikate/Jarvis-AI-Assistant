from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import json
import queue
import threading
import time
import logging
import argparse
from flask_cors import CORS
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    filename='jarvis_app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('jarvis_app')

# Load environment variables
load_dotenv()

# Set web UI mode before importing Jarvis
os.environ["WEB_UI_MODE"] = "true"

# Initialize flags and components
HAS_AUDIO = False
tts_queue = queue.Queue()
tts_active = threading.Event()
message_queue = queue.Queue()
transcription_queue = queue.Queue()

# Initialize TTS and audio components with fallbacks
try:
    logger.info("Initializing TTS components...")
    from TTS.api import TTS
    import numpy as np
    import sounddevice as sd
    
    # Initialize TTS
    tts_client = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
    TTS_SAMPLE_RATE = tts_client.synthesizer.output_sample_rate
    logger.info(f"TTS initialized with sample rate {TTS_SAMPLE_RATE}")
    
    # Test audio output
    try:
        with sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype='float32') as stream:
            logger.info("Audio output test successful")
            HAS_AUDIO = True
    except Exception as e:
        logger.error(f"Audio output test failed: {e}")
        HAS_AUDIO = False
except Exception as e:
    logger.error(f"Error initializing TTS: {e}")
    HAS_AUDIO = False

# Import remaining Jarvis components
try:
    from jarvis import (
        openai_client, whisper_model, weather_tool
    )
    logger.info("Jarvis components imported successfully")
except Exception as e:
    logger.error(f"Error importing Jarvis components: {e}")
    # Mock components if imports fail
    class MockClient:
        def __init__(self):
            self.chat = self  # Add chat attribute to fix the error
            
        def completions_create(self, **kwargs):
            class MockStream:
                def __iter__(self):
                    yield MockPart()
            return MockStream()
    
    class MockPart:
        class Delta:
            content = "I'm sorry, but I can't access the AI model right now."
        
        choices = [type('obj', (object,), {'delta': Delta()})]
    
    openai_client = MockClient()
    whisper_model = None
    
    class MockWeatherTool:
        def get_weather(self, loc):
            return f"Weather API not available. Can't check weather for {loc}."
    
    weather_tool = MockWeatherTool()

# TTS worker function
def tts_worker():
    logger.info("TTS worker thread started")
    if not HAS_AUDIO:
        logger.warning("Running without audio output")
        while True:
            try:
                phrase = tts_queue.get()
                if phrase is None:
                    break
                logger.info(f"Would speak (no audio): {phrase}")
                tts_active.set()
                # Simulate speaking time
                time.sleep(len(phrase) * 0.05)
                tts_active.clear()
            except Exception as e:
                logger.error(f"Error in TTS worker (no audio): {e}")
            time.sleep(0.1)
    else:
        logger.info("Running with audio output")
        try:
            with sd.OutputStream(
                samplerate=TTS_SAMPLE_RATE,
                channels=1,
                dtype='float32'
            ) as stream:
                logger.info("Audio output stream opened")
                while True:
                    try:
                        phrase = tts_queue.get()
                        if phrase is None:
                            break
                            
                        logger.info(f"TTS processing phrase: '{phrase}'")
                        
                        # Set flag before generating audio to allow UI to animate immediately
                        tts_active.set()
                        
                        # Generate speech
                        wav = tts_client.tts(phrase, speaker="p226")
                        logger.info(f"TTS generated {len(wav)} samples")
                        
                        # Play through audio device with precise start/stop animation signals
                        chunk_size = 1024
                        is_first_chunk = True
                        
                        # Play through audio device
                        for i in range(0, len(wav), chunk_size):
                            if not tts_active.is_set(): 
                                break
                                
                            # Process this audio chunk
                            current_chunk = np.asarray(wav[i:i+chunk_size], dtype=np.float32)
                            stream.write(current_chunk)
                            
                            # Check if it's the last chunk
                            if i + chunk_size >= len(wav) - chunk_size:
                                # Add a small delay before clearing the flag (smoother ending)
                                time.sleep(0.1)
                        
                        logger.info("TTS playback completed")
                        # Clear flag when done
                        tts_active.clear()
                    except queue.Empty:
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Error in TTS worker: {e}")
                        tts_active.clear()  # Make sure to clear the flag on error
        except Exception as e:
            logger.error(f"Error setting up audio stream: {e}")
            # Fall back to no audio mode
            logger.warning("Falling back to no-audio mode")
            while True:
                try:
                    phrase = tts_queue.get()
                    if phrase is None:
                        break
                    logger.info(f"Would speak (fallback): {phrase}")
                    tts_active.set()
                    # Simulate speaking time
                    time.sleep(len(phrase) * 0.05)
                    tts_active.clear()
                except Exception as e:
                    logger.error(f"Error in TTS worker fallback: {e}")
                time.sleep(0.1)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='frontend/build/static', 
            template_folder='frontend/build')

# Enable CORS for all routes
CORS(app)

# Routes
@app.route('/')
def index():
    logger.info("Index page requested")
    # Queue up a welcome message
    try:
        message = "Hello! I'm Jarvis. How can I help you today?"
        transcription_queue.put({"role": "assistant", "content": message})
        message_queue.put(message)
        tts_queue.put(message)
        logger.info("Welcome message queued")
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")
    return render_template('index.html')

@app.route('/api/speak', methods=['POST'])
def speak():
    logger.info("Speak endpoint called")
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            logger.warning("No text provided to speak endpoint")
            return jsonify({'error': 'No text provided'}), 400
        
        # Add the message to queues
        message_queue.put(text)
        transcription_queue.put({"role": "assistant", "content": text})
        tts_queue.put(text)
        
        logger.info(f"Message queued for speaking: '{text}'")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error in speak endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream-response')
def stream_response():
    logger.info("Stream response endpoint connected")
    def generate():
        while True:
            try:
                message = message_queue.get(timeout=1)
                if message:
                    logger.debug(f"Sending message to client: '{message}'")
                    # Send speaking start event
                    yield f"data: {json.dumps({'message': message, 'speaking': True, 'event': 'start'})}\n\n"
                    
                    # Add a small delay to ensure TTS has started
                    time.sleep(0.2)
                    
                    # Send periodic speaking updates while TTS is active
                    speaking_timeout = time.time() + len(message) * 0.1  # Fallback timeout
                    while tts_active.is_set() or time.time() < speaking_timeout:
                        if tts_active.is_set():
                            # Send speaking progress updates
                            yield f"data: {json.dumps({'event': 'speaking', 'speaking': True})}\n\n"
                            time.sleep(0.1)  # More frequent updates
                        else:
                            # If TTS ended before timeout, break early
                            break
                    
                    # Send speaking end event
                    yield f"data: {json.dumps({'event': 'end', 'speaking': False})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                time.sleep(1)
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/stream-transcription')
def stream_transcription():
    logger.info("Stream transcription endpoint connected")
    def generate():
        while True:
            try:
                message = transcription_queue.get(timeout=1)
                if message:
                    logger.debug(f"Sending transcription to client: {message}")
                    yield f"data: {json.dumps(message)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                time.sleep(1)
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    # Handle preflight CORS requests
    if request.method == 'OPTIONS':
        logger.info("Handling OPTIONS request for transcribe endpoint")
        return '', 204
        
    logger.info("Transcribe endpoint called")
    
    try:
        # Debug request info
        logger.debug(f"Request content type: {request.content_type}")
        logger.debug(f"Request files: {request.files}")
        logger.debug(f"Request headers: {request.headers}")
        
        if 'audio' not in request.files:
            logger.warning("No audio file provided")
            return jsonify({'error': 'No audio file provided', 'success': False}), 400
        
        audio_file = request.files['audio']
        logger.info(f"Received audio file: {audio_file.filename}")
        
        # Save the file
        audio_path = 'temp_audio.wav'
        audio_file.save(audio_path)
        logger.info(f"Audio file saved to {audio_path}")
        
        # Use whisper to transcribe if available
        if whisper_model:
            try:
                # Check if ffmpeg is available
                import subprocess
                try:
                    subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    ffmpeg_available = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("ffmpeg not found, using fallback transcription")
                    ffmpeg_available = False
                
                if ffmpeg_available:
                    result = whisper_model.transcribe(audio_path, temperature=0.0)
                    # Fix the key error with more robust handling
                    if isinstance(result, dict) and "segments" in result:
                        segments = result["segments"]
                        text = " ".join(seg.get("text", "").strip() for seg in segments) or ""
                    else:
                        # Direct text result
                        text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
                    
                    logger.info(f"Transcription result: {text}")
                else:
                    # Fallback message when ffmpeg is not available
                    text = "It feels half baked at the moment."
                    logger.warning(f"Using fallback text: {text}")
            except Exception as e:
                logger.error(f"Error transcribing audio: {str(e)}")
                text = "I couldn't understand that. Please try again."
        else:
            logger.warning("Whisper model not available")
            text = "Speech recognition is not available at the moment."
        
        # Add to transcription queue for UI
        if text:
            transcription_queue.put({"role": "user", "content": text})
        
        # Process text for wake words
        CALL_OUTS = ["hey jarvis", "hello jarvis"]
        low = text.lower()
        
        if any(cue in low for cue in CALL_OUTS):
            for cue in CALL_OUTS:
                if cue in low:
                    idx = low.find(cue) + len(cue)
                    prompt = text[idx:].strip().rstrip('?.!')
                    break
            
            # Process with GPT
            try:
                # First response while processing
                message_queue.put("Just a moment...")
                
                # Process with GPT
                buf, ends = "", {".", "?", "!"}
                full_response = ""
                
                # Check for weather query
                import re
                weather_match = re.search(r"\bweather\s+in\s+(.+?)(?:\s+(?:now|right now))?$", prompt, re.IGNORECASE)
                if weather_match:
                    loc = weather_match.group(1).strip()
                    response = weather_tool.get_weather(loc)
                    message_queue.put(response)
                    transcription_queue.put({"role": "assistant", "content": response})
                    tts_queue.put(response)
                    logger.info(f"Weather response for {loc}: {response}")
                else:
                    # Use OpenAI
                    try:
                        logger.info(f"Sending prompt to OpenAI: {prompt}")
                        # Fix the chat.completions_create usage
                        try:
                            if hasattr(openai_client, 'chat') and hasattr(openai_client.chat, 'completions'):
                                stream = openai_client.chat.completions.create(
                                    model="gpt-4o-mini", stream=True, max_tokens=150,
                                    messages=[
                                        {"role":"system","content":"You are Jarvis. Keep responses short and only use plain text formatting. Do not use markdown or special formatting."},
                                        {"role":"user","content":prompt}
                                    ]
                                )
                            else:
                                # Fall back to older API or mock
                                stream = openai_client.completions_create(
                                    model="gpt-4o-mini", stream=True, max_tokens=150,
                                    messages=[
                                        {"role":"system","content":"You are Jarvis. Keep responses short and only use plain text formatting. Do not use markdown or special formatting."},
                                        {"role":"user","content":prompt}
                                    ]
                                )
                        except Exception as api_err:
                            logger.error(f"Error calling OpenAI API: {api_err}")
                            # Mock response as fallback
                            class MockStream:
                                def __iter__(self):
                                    class MockDelta:
                                        content = "I'm sorry, I'm having trouble connecting to my AI brain right now."
                                    
                                    class MockChoice:
                                        def __init__(self):
                                            self.delta = MockDelta()
                                    
                                    part = type('obj', (object,), {'choices': [MockChoice()]})
                                    yield part
                            
                            stream = MockStream()
                        
                        for part in stream:
                            delta = getattr(part.choices[0].delta, "content", None)
                            if delta:
                                buf += delta
                                while any(buf.endswith(e) for e in ends):
                                    idx = max(buf.rfind(e) for e in ends) + 1
                                    segment = buf[:idx].strip()
                                    message_queue.put(segment)
                                    tts_queue.put(segment)
                                    full_response += segment + " "
                                    buf = buf[idx:]
                        
                        if buf.strip():
                            message_queue.put(buf.strip())
                            tts_queue.put(buf.strip())
                            full_response += buf.strip()
                        
                        # Add the full response to the transcription queue
                        if full_response:
                            transcription_queue.put({"role": "assistant", "content": full_response.strip()})
                            logger.info(f"Full response: {full_response.strip()}")
                    except Exception as e:
                        logger.error(f"Error with OpenAI: {e}")
                        error_msg = "I'm having trouble connecting to my AI brain right now."
                        message_queue.put(error_msg)
                        tts_queue.put(error_msg)
                        transcription_queue.put({"role": "assistant", "content": error_msg})
            except Exception as e:
                logger.error(f"Error processing prompt: {e}")
                error_msg = "I encountered an error processing your request."
                message_queue.put(error_msg)
                tts_queue.put(error_msg)
                transcription_queue.put({"role": "assistant", "content": error_msg})
        
        return jsonify({
            'transcription': text,
            'success': True
        })
    except Exception as e:
        logger.error(f"Unhandled exception in transcribe endpoint: {str(e)}")
        return jsonify({
            'error': f"Server error: {str(e)}",
            'success': False
        }), 500

@app.route('/api/status')
def status():
    """Endpoint to check system status"""
    return jsonify({
        'status': 'online',
        'audio_enabled': HAS_AUDIO,
        'whisper_available': whisper_model is not None,
        'openai_available': openai_client is not None
    })

@app.route('/api/tts-test')
def tts_test():
    """A simple endpoint to test TTS functionality"""
    logger.info("TTS test endpoint called")
    try:
        test_message = "This is a test message from Jarvis text-to-speech. Can you hear me?"
        tts_queue.put(test_message)
        message_queue.put(test_message)
        transcription_queue.put({"role": "assistant", "content": test_message})
        logger.info(f"TTS test message queued: '{test_message}'")
        return jsonify({
            'message': 'TTS test message sent', 
            'text': test_message,
            'audio_enabled': HAS_AUDIO
        })
    except Exception as e:
        logger.error(f"Error in TTS test endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start Jarvis AI Assistant')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the Flask server on')
    args = parser.parse_args()
    
    logger.info(f"Starting Jarvis AI Assistant on port {args.port}")
    print(f"Starting Jarvis AI Assistant (Audio: {'Enabled' if HAS_AUDIO else 'Disabled'})")
    
    # Start TTS worker thread
    try:
        logger.info("Starting TTS worker thread")
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()
        logger.info("TTS worker thread started successfully")
        
        # Send a startup message
        startup_msg = "Jarvis web interface is now online and ready."
        logger.info(f"Sending startup message: '{startup_msg}'")
        tts_queue.put(startup_msg)
        
        # Start Flask server
        logger.info(f"Starting Flask server on port {args.port}")
        app.run(debug=False, port=args.port, use_reloader=False, host='0.0.0.0')
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
