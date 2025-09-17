from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import json
import queue
import threading
import time
from dotenv import load_dotenv

# Set web UI mode before importing Jarvis
os.environ["WEB_UI_MODE"] = "true"

from jarvis import (
    tts_client, tts_queue, tts_active, openai_client, 
    whisper_model, mcp_client, weather_tool, ask_gpt,
    audio_queue, transcribe_meeting, CALL_OUTS, tts_worker
)

app = Flask(__name__, 
            static_folder='frontend/build/static', 
            template_folder='frontend/build')

message_queue = queue.Queue()
transcription_queue = queue.Queue()

# Track if Jarvis is speaking
jarvis_speaking = threading.Event()

@app.route('/')
def index():
    # When someone accesses the root, queue up a welcome message
    try:
        transcription_queue.put({"role": "assistant", "content": "Hello! I'm Jarvis. How can I help you today?"})
        message_queue.put("Hello! I'm Jarvis. How can I help you today?")
    except Exception as e:
        print(f"Error sending welcome message: {e}")
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
    
    # Process the text similar to transcribe_meeting function
    low = text.lower()
    response = None
    
    # Check for wake word
    if any(cue in low for cue in CALL_OUTS):
        for cue in CALL_OUTS:
            if cue in low:
                idx = low.find(cue) + len(cue)
                prompt = text[idx:].strip().rstrip('?.!')
                break
        
        # Process the prompt
        if mcp_client:
            server = mcp_client.choose(prompt)
            if server:
                response = mcp_client.call(server, prompt)
                message_queue.put(response)
        
        if not response:
            # Weather check
            import re
            m = re.search(r"\bin\s+(.+?)(?:\s+(?:now|right now))?$", prompt, re.IGNORECASE)
            if m:
                loc = m.group(1).strip()
                response = weather_tool.get_weather(loc)
                if re.match(r"(?i)(error|unable|weather api key|city not found)", response):
                    response = None
                else:
                    message_queue.put(response)
        
        # Fallback to GPT
        if not response:
            # Instead of directly using ask_gpt, we'll replicate its functionality
            jarvis_speaking.set()
            message_queue.put("Just give me a sec.")
            
            buf, ends = "", {".", "?", "!"}
            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini", stream=True, max_tokens=150,
                messages=[
                    {"role":"system","content":"You are Jarvis. Keep responses short."},
                    {"role":"user","content":prompt}
                ]
            )
            
            full_response = ""
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

if __name__ == '__main__':
    print("Starting Flask server...")
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting Flask server: {e}")
