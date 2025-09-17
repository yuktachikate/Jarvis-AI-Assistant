from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import json
import time
import threading
import queue
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='frontend/build/static', template_folder='frontend/build')
CORS(app)

# Global state
current_message = ""
is_speaking = False
message_queue = queue.Queue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'audio_enabled': True,
        'microphone_ready': True,
        'system_status': 'ready'
    })

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    global current_message, is_speaking
    
    # Simulate transcription
    demo_responses = [
        "Hello Jarvis, how are you today?",
        "What's the weather like?", 
        "Tell me a joke",
        "What time is it?",
        "How can you help me?",
        "Thank you Jarvis"
    ]
    
    import random
    user_message = random.choice(demo_responses)
    current_message = user_message
    
    # Add to message queue for streaming
    message_queue.put({'role': 'user', 'content': user_message})
    
    # Generate AI response
    ai_responses = {
        "Hello Jarvis, how are you today?": "Hello! I'm doing great, thank you for asking. I'm ready to assist you with anything you need.",
        "What's the weather like?": "I don't have access to real-time weather data in this demo, but I'd be happy to help you check the weather through other means!",
        "Tell me a joke": "Why don't scientists trust atoms? Because they make up everything! 😄",
        "What time is it?": f"The current time is {time.strftime('%I:%M %p')}",
        "How can you help me?": "I can help you with various tasks like answering questions, providing information, having conversations, and much more!",
        "Thank you Jarvis": "You're very welcome! I'm always here to help whenever you need assistance."
    }
    
    ai_response = ai_responses.get(user_message, "That's an interesting question! In this demo mode, I have limited capabilities, but I'd love to help you explore what I can do.")
    
    # Schedule AI response
    def delayed_response():
        time.sleep(1)
        global is_speaking
        is_speaking = True
        message_queue.put({'role': 'assistant', 'content': ai_response})
        message_queue.put({'event': 'start'})
        time.sleep(3)
        is_speaking = False
        message_queue.put({'event': 'end'})
    
    threading.Thread(target=delayed_response).start()
    
    return jsonify({'transcription': user_message})

@app.route('/api/stream-transcription')
def stream_transcription():
    def generate():
        while True:
            try:
                # Get message from queue with timeout
                message = message_queue.get(timeout=1)
                yield f"data: {json.dumps(message)}\n\n"
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
    
    return Response(generate(), mimetype='text/plain', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@app.route('/api/stream-response')
def stream_response():
    def generate():
        while True:
            try:
                if is_speaking:
                    yield f"data: {json.dumps({'speaking': True, 'event': 'speaking'})}\n\n"
                else:
                    yield f"data: {json.dumps({'speaking': False})}\n\n"
                time.sleep(0.5)
            except:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
    
    return Response(generate(), mimetype='text/plain', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@app.route('/api/tts-test')
def tts_test():
    global is_speaking
    is_speaking = True
    
    def test_speech():
        time.sleep(2)
        global is_speaking
        is_speaking = False
    
    threading.Thread(target=test_speech).start()
    return jsonify({'status': 'testing', 'message': 'Voice test initiated'})

if __name__ == '__main__':
    print("🚀 Starting Jarvis 3D Demo...")
    print("🎮 3D Interface with AI Character")
    print("🎤 Click and hold to speak (demo mode)")
    print("🌐 Opening at http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)