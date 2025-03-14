#!/bin/bash

# Set project root directory (adjusted for scripts/ location)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Path to virtual environment
VENV_PATH="$PROJECT_ROOT/.venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
  echo "Error: Virtual environment not found at $VENV_PATH"
  echo "Run ./scripts/init-venv.sh to create it."
  return 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate" and setting PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/client:$PROJECT_ROOT/server"

echo "PYTHONPATH set to: $PYTHONPATH"
echo "Virtual environment activated. 🚀"
