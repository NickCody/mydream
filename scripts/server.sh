#!/bin/bash

source scripts/activate-venv.sh
python3 server/main.py "$@" || echo "Exited"

