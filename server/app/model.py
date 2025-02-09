import sys
from app.config_loader import ModelConfigLoader
import json

# Get config name from command-line argument (default to first model)
CONFIG_PATH = "server/config.json"

try:
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Error loading config file: {e}")

config_name = sys.argv[1] if len(sys.argv) > 1 else next(iter(config_data.keys()), None)

if not config_name or config_name not in config_data:
    raise ValueError(f"❌ Model configuration '{config_name}' not found in {CONFIG_PATH}")

print(f"🔍 Selected model: {config_name}")

# Initialize and load the pipeline
config_loader = ModelConfigLoader(config_name)
PIPELINE = config_loader.initialize_pipeline()
FINAL_PIPELINE = config_loader.initialize_final_pipeline()