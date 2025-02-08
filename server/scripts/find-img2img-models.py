import os
import json

# Path to model directory
model_dir = "/Volumes/Dragan/StableDiffusion/models/Stable-diffusion/"

# Required components for StableDiffusionImg2ImgPipeline
REQUIRED_KEYS = {"vae", "text_encoder", "tokenizer", "unet", "scheduler", "safety_checker", "feature_extractor"}

# Find all JSON config files
configs = [f for f in os.listdir(model_dir) if f.endswith(".json")]

compatible_models = []

for config_file in configs:
    config_path = os.path.join(model_dir, config_file)
    
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Check if the config contains all required keys
        if REQUIRED_KEYS.issubset(config_data.keys()):
            model_name = config_file.replace(".json", "")
            compatible_models.append(model_name)
            print(f"✅ Compatible: {model_name}")
        else:
            print(f"❌ Incompatible: {config_file} (missing {REQUIRED_KEYS - set(config_data.keys())})")
    
    except Exception as e:
        print(f"⚠️ Error reading {config_file}: {e}")

# Summary
if compatible_models:
    print("\n🎯 Compatible Models:", compatible_models)
else:
    print("\n❌ No compatible models found.")