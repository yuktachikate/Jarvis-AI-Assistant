#!/bin/bash

# Build script for Jarvis UI
echo "Building React app for Jarvis AI Assistant..."

# Move to frontend directory
cd "$(dirname "$0")/frontend"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install
  echo "Dependencies installed"
else
  echo "Frontend dependencies already installed"
fi

# Build the React app
echo "Building React application..."
npm run build

# Move back to project root
cd ..

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
  echo "Creating .env file..."
  echo "WEB_UI_MODE=true" > .env
  echo "# Add your OpenAI API key here" >> .env
  echo "# OPENAI_API_KEY=your_key_here" >> .env
  echo ".env file created"
else
  echo ".env file already exists"
fi

echo "Done! You can now run the Flask app with 'python unified_app.py'"
echo "Access the UI at http://localhost:5001"
