from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import json
import queue
import threading
import time

# Set up Flask app
app = Flask(__name__, 
            static_folder='frontend/build/static', 
            template_folder='frontend/build')

# Set up queues for messages
message_queue = queue.Queue()
transcription_queue = queue.Queue()

# Routes
@app.route('/')
def index():
    print("Index page requested")
    # Queue up a welcome message
    try:
        message = "Hello! I'm Jarvis. How can I help you today?"
        transcription_queue.put({"role": "assistant", "content": message})
        message_queue.put(message)
        print("Welcome message queued")
    except Exception as e:
        print(f"Error sending welcome message: {e}")
    return render_template('index.html')

@app.route('/api/speak', methods=['POST'])
def speak():
    print("Speak endpoint called")
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            print("No text provided to speak endpoint")
            return jsonify({'error': 'No text provided'}), 400
        
        # Add the message to both queues
        message_queue.put(text)
        transcription_queue.put({"role": "assistant", "content": text})
        print(f"Message queued for speaking: '{text}'")
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error in speak endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream-response')
def stream_response():
    print("Stream response endpoint connected")
    def generate():
        while True:
            try:
                message = message_queue.get(timeout=1)
                if message:
                    print(f"Sending message to client: '{message}'")
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
    print("Stream transcription endpoint connected")
    def generate():
        while True:
            try:
                message = transcription_queue.get(timeout=1)
                if message:
                    print(f"Sending transcription to client: {message}")
                    yield f"data: {json.dumps(message)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                time.sleep(1)
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    print("Transcribe endpoint called")
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    # Simulate transcription
    transcription = "This is a simulated transcription since we're not doing real speech recognition in this simplified version"
    
    # Add to transcription queue for UI
    transcription_queue.put({"role": "user", "content": transcription})
    
    # Simulate a response
    message_queue.put("I heard you, but I'm just a simplified version without real speech recognition.")
    transcription_queue.put({"role": "assistant", "content": "I heard you, but I'm just a simplified version without real speech recognition."})
    
    return jsonify({
        'transcription': transcription,
        'success': True
    })

if __name__ == '__main__':
    print("Starting Flask server without audio...")
    try:
        # Start Flask server
        app.run(debug=True, port=5000, use_reloader=False)
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
