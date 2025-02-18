#!/bin/bash

set -eou pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pip install -c $PROJECT_ROOT/constraints.txt transformers diffusers["torch"] tf-keras==2.17.0 accelerate
pip install -c $PROJECT_ROOT/constraints.txt torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall
