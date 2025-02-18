@echo off

pip install 'numpy<2' torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall
