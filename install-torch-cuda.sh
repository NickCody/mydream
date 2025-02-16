#!/bin/bash

set -eou pipefail

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall
