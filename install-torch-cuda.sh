#!/bin/bash

set -eou pipefail

pip install numpy<2 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall
