from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
import os
# Define model paths
checkpoint_path = "/Volumes/Dragan/StableDiffusion/models/Stable-diffusion/nekorayxl_v06W3.safetensors"
output_dir = "/Volumes/Dragan/StableDiffusion/models/Diffusers_NekorayXL"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📂 Created output directory: {output_dir}")
    
# Corrected function call
download_from_original_stable_diffusion_ckpt(
    checkpoint_path,
    output_dir,
    extract_ema=True,
    from_safetensors=True  # Important for .safetensors models
)

print("✅ Model converted to diffusers format!")