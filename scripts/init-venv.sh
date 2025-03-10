#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

echo "NOTE: Using Python interpreter: $(which python3)"

if [[ "$OSTYPE" =~ [dD]arwin(64)? ]]; then
    DARWIN_FOUND=1
else
    DARWIN_FOUND=0
fi

if [[ $DARWIN_FOUND -eq 1 ]]; then
  export CPATH="$(brew --prefix portaudio)/include"
  export LIBRARY_PATH="$(brew --prefix portaudio)/lib"
fi

#
# Set project root directory (assumed to be the script's location)
#
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

#
# Create Virtual Environment
#
echo "Creating virtual environment in ..."
VENV_PATH="$PROJECT_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
  rm -rf "$VENV_PATH"
fi
python3 -m venv "$VENV_PATH"

#
# Activate the virtual environment
#
echo "Activating virtual environment ..."
source "$VENV_PATH/bin/activate"

pip3 install --upgrade pip

#
# Linux/Cuda
#
if [[ $DARWIN_FOUND -eq 0 ]]; then
  $PROJECT_ROOT/scripts/install-torch-cuda.sh 
fi

#
# requirements.txt
#
pip3 install  -c constraints.txt -r "$PROJECT_ROOT/requirements.txt"

#
# GUI requirements
#
if [[ $DARWIN_FOUND -eq 1 ]]; then
    pip install -c constraints.txt -r gui-requirements.txt
fi

echo "Success! Activate the venv and PYTHONPATH via: `source scripts/activate-venv.sh`"
