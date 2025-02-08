from huggingface_hub import hf_hub_download, try_to_load_from_cache
import os

# Print Hugging Face cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
print("Hugging Face cache directory:", cache_dir)

# Find a specific model's location
model_path = try_to_load_from_cache("stabilityai/stable-diffusion-xl-base-1.0")
print("Model cached at:", model_path)