from PIL import Image, ImageOps
from app.model import PIPELINE, config_loader
import math
import numpy as np
from diffusers import DPMSolverMultistepScheduler
from app.codeformer_api import enhance_faces
import cv2

def round_to_multiple(value, multiple=64):
    return math.ceil(value / multiple) * multiple

def apply_codeformer(codeformer_cfg, image):
    print(f"codeformer_cfg: {codeformer_cfg}")
    # Retrieve CodeFormer settings
    codeformer_enabled = codeformer_cfg.get("enabled", False)
    
    # ✅ Apply CodeFormer enhancement if enabled
    if codeformer_enabled:
        codeformer_weight = codeformer_cfg.get("weight", 0.7)
        codeformer_upscale = codeformer_cfg.get("upscale", 1)
        codeformer_has_aligned = codeformer_cfg.get("has_aligned", False)
        codeformer_paste_back = codeformer_cfg.get("paste_back", True)
        print(f"\n✨ Applying CodeFormer face enhancement (weight={codeformer_weight})...")
        return enhance_faces(
            config_loader.device,
            image,
            enabled=codeformer_enabled,
            weight=codeformer_weight,
            upscale=codeformer_upscale,
            has_aligned=codeformer_has_aligned,
            paste_back=codeformer_paste_back
        )
    else:
        print("\n🚫 CodeFormer enhancement skipped.")
        return image

def apply_gaussian_blur(image: Image.Image, ksize=15, sigma=0):
    """
    Applies a Gaussian blur using OpenCV for hardware acceleration.

    Parameters:
    - image (PIL.Image): The input image.
    - ksize (int): Kernel size (must be an odd number, e.g., 15).
    - sigma (float): Standard deviation (0 = auto).

    Returns:
    - PIL.Image: The blurred image.
    """
    # Convert PIL Image to OpenCV (NumPy array)
    image_cv = np.array(image)  # RGB order

    # Convert RGB to BGR (OpenCV uses BGR)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # ✅ Apply Gaussian Blur (Hardware Accelerated)
    blurred_cv = cv2.GaussianBlur(image_cv, (ksize, ksize), sigma)

    # Convert back to RGB
    blurred_rgb = cv2.cvtColor(blurred_cv, cv2.COLOR_BGR2RGB)

    # Convert back to PIL image
    return Image.fromarray(blurred_rgb)

def transform_image(input_image: Image.Image, prompt: str, bg_prompt: str, mask: Image.Image = None) -> Image.Image:
    """
    Transforms the input image based on the provided prompt.
    Supports inpainting if a mask is provided and applies CodeFormer if enabled.
    
    - White areas in the mask are preserved.
    - Black areas in the mask are AI-generated.
    """
    
    # ✅ Call the pipeline with or without a mask
    if mask is None:
        raise ValueError("Mask is required for inpainting mode.")
    
    # Retrieve parameters from config
    params = config_loader.get_parameters()
    codeformer_config = config_loader.config_entry.get("codeformer", {})
   
    standard_size = (round_to_multiple(params.get("width", 640)), round_to_multiple(params.get("height", 512)))
    
    # Standardize on generated size 
    mask = mask.resize(standard_size, Image.LANCZOS)
    bg_mask = ImageOps.invert(mask)
    input_image = input_image.resize(standard_size, Image.LANCZOS)
    
    print(f"🎭 Running inpainting for background with {prompt}, parameters: \n{params}")
    result = PIPELINE(
        prompt=prompt,
        width=round_to_multiple(params.get("width", 640)),
        height=round_to_multiple(params.get("height", 512)),
        negative_prompt=params.get("negative_prompt", ""),
        guidance_scale=params.get("guidance_scale", 4),
        num_inference_steps=params.get("num_inference_steps", 20),
        image=input_image,
        mask_image=mask,  # Pass the mask for inpainting
        strength=params.get("strength", 0.2)
    )
   
    foreground_image = result.images[0] 
    foreground_image = apply_codeformer(codeformer_config, foreground_image)
    print(f"Foreground image: {type(foreground_image)}: {foreground_image.size[0]}x{foreground_image.size[1]}")
    
    bg_params = config_loader.get_bg_parameters()
    print(f"🎭 Running inpainting for background with {bg_prompt}, parameters: \n{bg_params}")
    result = PIPELINE(
        prompt=bg_prompt,
        width=round_to_multiple(bg_params.get("width", 640)),
        height=round_to_multiple(bg_params.get("height", 512)),
        negative_prompt=bg_params.get("negative_prompt", ""),
        guidance_scale=bg_params.get("guidance_scale", 4),
        num_inference_steps=bg_params.get("num_inference_steps", 20),
        image=input_image,
        mask_image=bg_mask,  # Pass the mask for inpainting
        strength=bg_params.get("strength", 0.2)
    )
    
    background_image = result.images[0]
    # Apply blur to background_image
    if (params.get("blur", None) is not None):
        background_image = apply_gaussian_blur(background_image, ksize=params.get("blur", 15), sigma=0)
    print(f"Background image: {type(background_image)}: {background_image.size[0]}x{background_image.size[1]}")

    print(f"Mask image: {type(mask)}: {mask.size[0]}x{mask.size[1]}")
    scaled_mask = mask.resize(foreground_image.size, Image.LANCZOS)
    final_image = Image.composite(foreground_image, background_image, scaled_mask)
    
    # ✅ Print generated image details after CodeFormer processing
    print("\n📸 Output Image Details (After CodeFormer Enhancement):")
    print(f"Final Image Size: {final_image.size[0]}x{final_image.size[1]}")
    np_image = np.array(final_image)
    print(f"Image Stats - Min: {np_image.min()}, Max: {np_image.max()}, Mean: {np_image.mean()}")

    return final_image 