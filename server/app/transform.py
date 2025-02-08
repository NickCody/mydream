from PIL import Image, ImageOps
from app.model import PIPELINE, config_loader
import math
import numpy as np
from diffusers import DPMSolverMultistepScheduler
import os, sys
from app.codeformer_api import enhance_faces

def round_to_multiple(value, multiple=64):
    return math.ceil(value / multiple) * multiple

def transform_image(input_image: Image.Image, prompt: str, mask: Image.Image = None) -> Image.Image:
    """
    Transforms the input image based on the provided prompt.
    Supports inpainting if a mask is provided and applies CodeFormer if enabled.
    
    - White areas in the mask are preserved.
    - Black areas in the mask are AI-generated.
    """

    # Retrieve parameters from config
    params = config_loader.get_parameters()
    strength = params.get("strength", 0.2)
    num_inference_steps = params.get("num_inference_steps", 20)
    guidance_scale = params.get("guidance_scale", 4)
    width = round_to_multiple(params.get("width", 512))
    height = round_to_multiple(params.get("height", 512))
    negative_prompt = params.get("negative_prompt", "")

    # Retrieve CodeFormer settings
    codeformer_config = params.get("codeformer", {})
    codeformer_enabled = codeformer_config.get("enabled", True)
    codeformer_weight = codeformer_config.get("weight", 0.7)
    codeformer_upscale = codeformer_config.get("upscale", 1)
    codeformer_has_aligned = codeformer_config.get("has_aligned", False)
    codeformer_paste_back = codeformer_config.get("paste_back", True)

    print(f"Rendering image with prompt: {prompt}")
    print(f"width: {width}, height: {height}, num_steps: {num_inference_steps}")

    # ✅ Ensure image is in the correct format and size
    input_image = input_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

    # ✅ Process mask if provided
    if mask is not None:
        print("🔹 Detected mask: Resizing and converting to grayscale")
        mask = mask.convert("L").resize((width, height), Image.Resampling.LANCZOS)
      
        mask = mask.point(lambda p: 255 if p > 127 else 0)
        print("✅ Mask forced to pure black & white") 
        
        # Save for visual inspection
        mask.save("temp/debug_mask.png")
        print("Saved debug mask as debug_mask.png for verification")

        # Debugging: Check pixel values in the mask
        mask_np = np.array(mask)
        print(f"Mask Debug - Min: {mask_np.min()}, Max: {mask_np.max()}, Mean: {mask_np.mean()}")
    
    # ✅ Call the pipeline with or without a mask
    if mask is not None:
        print("🎭 Running inpainting mode...")
        result = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            image=input_image,
            mask_image=mask,  # Pass the mask for inpainting
            strength=strength
        )
    else:
        print("🎨 Running regular img2img mode...")
        result = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            image=input_image,
            strength=strength
        )

    # Get generated image
    generated_image = result.images[0]

    # ✅ Apply CodeFormer enhancement if enabled
    if codeformer_enabled:
        print(f"\n✨ Applying CodeFormer face enhancement (weight={codeformer_weight})...")
        enhanced_image = enhance_faces(
            config_loader.device,
            generated_image,
            enabled=codeformer_enabled,
            weight=codeformer_weight,
            upscale=codeformer_upscale,
            has_aligned=codeformer_has_aligned,
            paste_back=codeformer_paste_back
        )
    else:
        print("\n🚫 CodeFormer enhancement skipped.")
        enhanced_image = generated_image  # Use the original output

    # Print formatted parameter details
    print("\n🎨 Generation Parameters:")
    print(f"  - Strength: {strength}")
    print(f"  - Inference Steps: {num_inference_steps}")
    print(f"  - Guidance Scale: {guidance_scale}")
    print(f"  - Width: {width}")
    print(f"  - Height: {height}")
    print(f"  - Negative Prompt: {negative_prompt if negative_prompt else 'None'}")
    print(f"  - Inpainting Mode: {'Enabled' if mask is not None else 'Disabled'}")

    # ✅ Print generated image details after CodeFormer processing
    print("\n📸 Output Image Details (After CodeFormer Enhancement):")
    print(f"Final Image Size: {enhanced_image.size[0]}x{enhanced_image.size[1]}")
    np_image = np.array(enhanced_image)
    print(f"Image Stats - Min: {np_image.min()}, Max: {np_image.max()}, Mean: {np_image.mean()}")

    # Save debug images
    generated_image.save("temp/debug_generated_image.png")
    enhanced_image.save("temp/debug_enhanced_image.png")
    print("✅ Saved generated image as debug_generated_image.png")
    print("✅ Saved enhanced image as debug_enhanced_image.png")

    return enhanced_image  # Return final enhanced image