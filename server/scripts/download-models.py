import os
import json
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files

# Load API Key from Environment
HF_TOKEN = os.getenv("HF_API_KEY")

if not HF_TOKEN:
    raise RuntimeError("❌ Hugging Face API key not found. Set HF_API_KEY environment variable.")

# Load Hugging Face cache directory
HF_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(f"📂 Using Hugging Face cache directory: {HF_CACHE}")

# Load Config
CONFIG_PATH = "server/config.json"

try:
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Error loading config file: {e}")

def download_model(config_name, config):
    """Download model weights and configs to the Hugging Face cache directory."""
    print(f"\n🔽 Downloading model: {config_name} ({config['model_name']})")

    if config["type"] == "huggingface":
        repo_id = config["model_name"]
        checkpoint_file = config["checkpoint_file"]

        # List available files
        available_files = list_repo_files(repo_id, token=HF_TOKEN)
        print(f"📜 Available files: {available_files}")

        # Check if checkpoint file exists
        if checkpoint_file not in available_files:
            print(f"⚠️ Warning: {checkpoint_file} not found in {repo_id}. Trying to auto-detect latest model file.")
            checkpoint_file = next((f for f in available_files if f.endswith(".safetensors")), None)

        if not checkpoint_file:
            print(f"❌ No suitable checkpoint found for {repo_id}. Skipping.")
            return

        # Download checkpoint file into HF_HOME cache
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_file,
            cache_dir=HF_CACHE,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
        print(f"✅ Model cached at: {checkpoint_path}")

        # Download config files into HF_HOME cache
        config_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=HF_CACHE,
            allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"],
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
        print(f"✅ Config files cached at: {config_path}")

    elif config["type"] == "local":
        print(f"📂 Skipping local model: {config_name} (Stored at {config['weights_path']})")

    else:
        print(f"❌ Unknown model type for {config_name}: {config['type']}")

# Iterate through config and download each model
for model_name, model_config in config_data.items():
    download_model(model_name, model_config)

print("\n🎉 All models are now cached in HF_HOME!")