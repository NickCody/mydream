#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Make sure you do these two things first:
# 1. Install Python 3.12.8 using pyenv.
# 2. Install portaudio using Homebrew.

# MacOS
if [ "$(uname)" == "Darwin" ]; then
  export CPATH="$(brew --prefix portaudio)/include"
  export LIBRARY_PATH="$(brew --prefix portaudio)/lib"
fi

# Set project root directory (assumed to be the script's location)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use the environment variable PYTHON_BIN if set, otherwise default to /Users/nic/.pyenv/versions/3.12.8/bin/python3
PYTHON_BIN=${PYTHON_BIN:-/Users/nic/.pyenv/versions/3.12.8/bin/python3}

echo "Using Python interpreter: $PYTHON_BIN"

# Create the virtual environment in the project root
VENV_PATH="$PROJECT_ROOT/.venv"

echo "Creating virtual environment in $VENV_PATH ..."
if [ -d "$VENV_PATH" ]; then
  rm -rf "$VENV_PATH"
fi
"$PYTHON_BIN" -m venv "$VENV_PATH"
if [ $? -ne 0 ]; then
  echo "Failed to create virtual environment."
  exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment ..."
source "$VENV_PATH/bin/activate"

pip3 install --upgrade pip

# Check if a unified requirements.txt exists
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: $REQUIREMENTS_FILE not found."
  deactivate
  exit 1
fi

echo "Installing dependencies from $REQUIREMENTS_FILE ..."
pip3 install -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
  echo "Failed to install dependencies."
  deactivate
  exit 1
fi

# Set PYTHONPATH to include both client and server
echo "Setting PYTHONPATH for both client/ and server/..."
export PYTHONPATH="$PROJECT_ROOT/client:$PROJECT_ROOT/server:$PROJECT_ROOT/CodeFormer"

# Fix CodeFormer
source $VENV_PATH/bin/activate
echo "Installing CodeFormer dependencies ..."
cd CodeFormer
python ./basicsr/setup.py install

echo "Virtual environment setup complete."
echo "To activate it manually, run: source $VENV_PATH/bin/activate"