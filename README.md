# Jarvis AI Assistant

A voice-enabled AI assistant with a dog animation UI. Jarvis listens to your voice commands, responds with AI-generated answers, and features a cute animated dog face that moves when speaking.

## Features

- **Voice Recognition**: Capture and transcribe spoken commands
- **AI Responses**: Get intelligent responses powered by OpenAI's models
- **Animated Dog UI**: An animated dog face that speaks when Jarvis responds
- **Weather Information**: Ask about the weather in any location
- **Text-to-Speech**: Hear Jarvis speak the responses (when audio output is available)
- **Modern Chat Interface**: Clean, mobile-friendly UI

## Quick Start

The easiest way to get started is to use the run script:

```bash
# Make the script executable
chmod +x run.sh

# Run the app
./run.sh
```

This will:
1. Create a Python virtual environment if needed
2. Install all required dependencies
3. Build the React frontend if not already built
4. Start the Jarvis web application

Then open your browser to [http://localhost:5001](http://localhost:5001) to use Jarvis.

## Manual Setup

If you prefer to set up manually:

1. **Setup Python environment**
   ```bash
   # Create and activate virtual environment
   python -m venv aibot
   source aibot/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Build the React frontend**
   ```bash
   # Make the build script executable
   chmod +x build.sh
   
   # Run the build script
   ./build.sh
   ```

3. **Configure environment variables**
   Create a `.env` file with:
   ```
   WEB_UI_MODE=true
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   python unified_app.py
   ```

## Using Jarvis

1. Open your browser to [http://localhost:5000](http://localhost:5000)
2. Click and hold the "Hold to Speak" button while speaking
3. Release the button when done speaking
4. Wait for Jarvis to process your request and respond

Example commands:
- "Hey Jarvis, what's the weather in New York?"
- "Hello Jarvis, tell me a joke."
- "Hey Jarvis, what time is it?"

## Troubleshooting

### Audio Output Issues

If you're not hearing Jarvis speak:

1. Check that your system audio is working properly
2. Look for errors in the `jarvis_app.log` file
3. The app will automatically fall back to a mode without audio output if it detects issues

### Microphone Access

- Make sure to grant microphone access to your browser
- Check browser console for any permission errors

### OpenAI API Key

If you're getting errors about the OpenAI API:
1. Make sure you've added your API key to the `.env` file
2. Check that your API key is valid and has sufficient quota

## Available Application Versions

Several versions of the application are available for different purposes:

- `unified_app.py` - The main application with audio fallbacks (recommended)
- `noaudio_app.py` - Version without audio functionality (for UI testing)
- `direct_tts_test.py` - Test script for the text-to-speech functionality
- `simple_app.py` - Simplified version for debugging

## Development

To contribute or modify:

1. Make changes to the frontend code in the `frontend/src` directory
2. Run `./build.sh` to rebuild the frontend
3. Modify the backend code in `unified_app.py` or `jarvis.py`
4. Run `python unified_app.py` to test your changes

## License

This project is open-source and available under the MIT License.

## Development

- To work on the React frontend in development mode:
  ```bash
  cd frontend
  npm start
  ```
  This will run the frontend on port 3000 with hot reloading.

- To run only the Flask backend:
  ```bash
  python app.py
  ```

## Usage

1. Click and hold the "Hold to Speak" button
2. Say "Hey Jarvis" followed by your question
3. Release the button to send the recording
4. Watch the dog animation as Jarvis responds

## Features

- Voice-activated AI assistant
- Dog animation that speaks with Jarvis's responses
- Clean, modern UI
- Weather information capability
- Integration with OpenAI's GPT models
