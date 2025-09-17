#!/bin/bash

# Launcher script for Jarvis AI Assistant
echo "Starting Jarvis AI Assistant..."

# Check if virtual environment exists, create if not
if [ ! -d "aibot" ]; then
  echo "Creating Python virtual environment..."
  python -m venv aibot
  source aibot/bin/activate
  echo "Installing dependencies..."
  pip install -r requirements.txt
else
  echo "Activating virtual environment..."
  source aibot/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
  echo "Creating .env file..."
  echo "WEB_UI_MODE=true" > .env
  
  # Prompt for OpenAI API key
  read -p "Enter your OpenAI API key (or leave blank to skip): " api_key
  if [ ! -z "$api_key" ]; then
    echo "OPENAI_API_KEY=$api_key" >> .env
    echo "API key saved to .env file"
  else
    echo "# OPENAI_API_KEY=your_key_here" >> .env
    echo "No API key provided, add it later to .env file"
  fi
fi

# Check if frontend is built
if [ ! -d "frontend/build" ]; then
  echo "Frontend not built yet. Building now..."
  ./build.sh
else
  echo "Frontend already built"
fi

# Run the application
echo "Starting Jarvis AI Assistant..."
python unified_app.py

echo "Jarvis is now running! Access it at http://localhost:5001"
