import os
import json
import torch
import sys
from diffusers import DiffusionPipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config_loader import config_data  # Load models from config.json

def get_model_properties(model_name):
    """Fetches key properties of a Stable Diffusion model."""
    try:
        print(f"\n🔄 Loading model: {model_name} ...")
        
        # Load pipeline
        pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

        # Extract properties
        properties = {
            "model_name": model_name,
            "pipeline_class": type(pipeline).__name__,
            "text_encoder": type(pipeline.text_encoder).__name__,
            "tokenizer_max_tokens": pipeline.tokenizer.model_max_length,
            "scheduler": type(pipeline.scheduler).__name__,
            "unet_config": pipeline.unet.config,
            "vae_config": pipeline.vae.config,
        }

        return properties

    except Exception as e:
        print(f"❌ Failed to query model {model_name}: {e}")
        return None

def print_model_info():
    """Loops through config.json and prints properties for each model."""
    
    print("\n🔍 Enumerating Models from config.json...\n")

    for model_key, model_data in config_data.items():
        model_name = model_data["model_name"]

        print("=" * 80)
        print(f"📌 Model: {model_key}")
        print("=" * 80)

        # 🔹 Section 1: Model Properties
        print("\n🔬 Model Properties:")
        model_props = get_model_properties(model_name)
        
        if model_props:
            for key, value in model_props.items():
                if isinstance(value, dict):  # Print dict configs as formatted JSON
                    print(f"  - {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"      {sub_key}: {sub_value}")
                else:
                    print(f"  - {key}: {value}")

        # 🔹 Section 2: Config Properties
        print("\n⚙️ Config Properties (from config.json):")
        for key, value in model_data.items():
            if key != "model_name":  # Skip model_name since it's already printed
                if isinstance(value, dict):
                    print(f"  - {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"      {sub_key}: {sub_value}")
                else:
                    print(f"  - {key}: {value}")

        print("\n" + "-" * 80 + "\n")

# Run script
if __name__ == "__main__":
    print_model_info()