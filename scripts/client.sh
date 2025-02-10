#!/bin/bash

source scripts/activate-venv.sh
python -m main "$@" || echo "Exited"

