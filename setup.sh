#!/bin/bash

# Script to install dependencies for Jarvis AI Assistant
echo "Installing dependencies for Jarvis AI Assistant..."

# Activate virtual environment if it exists
if [ -d "aibot" ]; then
    source aibot/bin/activate
    echo "Activated virtual environment"
else
    echo "Creating new virtual environment..."
    python -m venv aibot
    source aibot/bin/activate
    echo "Virtual environment created and activated"
fi

# Install required packages
echo "Installing required packages..."
pip install flask python-dotenv openai-whisper webrtcvad numpy sounddevice TTS requests anyio mcp

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY environment variable not set."
    echo "Would you like to set it now? (y/n)"
    read answer
    if [ "$answer" = "y" ]; then
        echo "Enter your OpenAI API key:"
        read api_key
        echo "OPENAI_API_KEY=$api_key" >> .env
        echo "API key added to .env file"
    else
        echo "Please set OPENAI_API_KEY before running the application"
    fi
else
    echo "OPENAI_API_KEY environment variable is set"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "WEB_UI_MODE=true" > .env
    echo "# Add your OpenAI API key here if not set in your environment" >> .env
    echo "# OPENAI_API_KEY=your_key_here" >> .env
    echo ".env file created"
fi

# Build the React frontend
echo "Building React frontend..."
cd frontend && npm install && npm run build && cd ..

echo "Setup complete! You can now run the application with:"
echo "python new_app.py"
