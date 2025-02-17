# RUMPLE "MY DREAM"

NOTE: This README is a work in progress.

## Install from Github

```bash
bash <(curl -sSL https://raw.githubusercontent.com/NickCody/mydream/main/install-mydream.sh)
```

## INSTALL PREREQUISITES

Most of these instructions are for the MacOS on Apple Silicon. Windows instructions will be added later.

Some preliminary installations, you'll need Homebrew, python3, and portaudio.

```bash
brew install pyenv
pyenv global 3.12.0
brew install python3.12 # 10 and 11 are probably ok
brew install portaudio
brew install cmake
python3 -m pip install jax-metal
```

On H200 GPU's, you need to run `nvidia-smi` and note the CUDA Version:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.08             Driver Version: 550.127.08     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GH200 480GB             On  |   00000000:DD:00.0 Off |                    0 |
| N/A   39C    P0            253W /  700W |    9446MiB /  97871MiB |     14%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```

And run the appropriate torch install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall
```

### SPEECH RECOGNITION SETUP

Download model to client/audio/models:

- <https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip>
- <https://alphacephei.com/vosk/models>

## Run

You need to run the client and the server in two separate terminals. Before running each, you need to install the dependencies:

```bash
scripts/init-venv.sh
```

NOTE: To run the GUI, additionally you need to `pip install gui-requirements.txt`

```bash
Then activate the `venv`::

```bash
source scripts/activate-venv.sh
```

Then you can run both of these, except in separate terminals:

```bash
scripts/client.sh
scripts/server.sh [model-name] # optpional model name
```

Typical config looks like this:

```json
    "xl": {
        "model_name": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "pipeline_class": "AutoPipelineForImage2Image",
        "parameters": {
            "strength": 0.33,                   # 0 incoming image is strong, 1 weak
            "num_inference_steps": 20,          # Lower for speed, higher for quality          
            "guidance_scale": 12.0,             # How strongly to follow prompt
            "width": 640,                       
            "height": 512,
            "negative_prompt": "painting, anime, illustration"
        },
        "scheduler": {
            "type": "EulerAncestralDiscreteScheduler"
        }
    }
```
## Running

On Windows/Mac/Linux, you need to set:

```bash
export HF_API_KEY=<your key>
export SAFETENSOR_HOME=<safetensor directory>
```

## NOTES

- Configs for server-side image generation models are in `server/config.json`
- Default prompt is in `client/gui/main_window.py`, lame but will fix later

keys in bashrc
comment out pyqt and
