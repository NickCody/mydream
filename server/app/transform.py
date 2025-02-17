from PIL import Image, ImageOps
from app.model import PIPELINE, FINAL_PIPELINE, config_loader
import math
import numpy as np
from diffusers import DPMSolverMultistepScheduler
import cv2
import time
import torch

def round_to_multiple(value, multiple=64):
    return math.ceil(value / multiple) * multiple

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

def transform_image(input_image: Image.Image, prompt: str, bg_prompt: str, processed_width: int, processed_height: int, mask: Image.Image = None) -> Image.Image:
    """
    Transforms the input image based on the provided prompt.
    Supports inpainting if a mask is provided 
    
    - White areas in the mask are preserved.
    - Black areas in the mask are AI-generated.
    """
    
    # ✅ Call the pipeline with or without a mask
    if mask is None:
        raise ValueError("Mask is required for inpainting mode.")
    
    # Retrieve parameters from config
    params = config_loader.get_parameters()
   
    standard_size = (processed_width, processed_height)
    
    # Standardize on generated size 
    mask = mask.resize(standard_size, Image.LANCZOS)
    bg_mask = ImageOps.invert(mask)
    input_image = input_image.resize(standard_size, Image.LANCZOS)
    
    print(f"🎭 Running inpainting for background with {prompt}, parameters: \n{params}")
    result = PIPELINE(
        prompt=prompt,
        width=processed_width,
        height=processed_height,
        negative_prompt=params.get("negative_prompt", ""),
        guidance_scale=params.get("guidance_scale", 4),
        num_inference_steps=params.get("num_inference_steps", 20),
        image=input_image,
        mask_image=mask,  # Pass the mask for inpainting
        strength=params.get("strength", 0.2)
    )
   
    foreground_image = result.images[0] 
    print(f"Foreground image: {type(foreground_image)}: {foreground_image.size[0]}x{foreground_image.size[1]}")
    
    bg_params = config_loader.get_bg_parameters()
    print(f"🎭 Running inpainting for background with {bg_prompt}, parameters: \n{bg_params}")
    result = PIPELINE(
        prompt=bg_prompt,
        width=processed_width,
        height=processed_height,
        negative_prompt=bg_params.get("negative_prompt", ""),
        guidance_scale=bg_params.get("guidance_scale", 4),
        num_inference_steps=bg_params.get("num_inference_steps", 20),
        image=input_image,
        mask_image=bg_mask,  # Pass the mask for inpainting
        strength=bg_params.get("strength", 0.2)
    )
    
    background_image = result.images[0]
    
    try:
        # Apply blur to background_image
        if (bg_params.get("blur", None) is not None):
            print(f"Applying blur to background image with ksize={bg_params.get('blur', 15)}")
            blurred_background_image = apply_gaussian_blur(background_image, ksize=bg_params.get("blur", 15), sigma=0)
            background_image = blurred_background_image
        else:
            print("Skipping blur for background image")
    except Exception as e:
        print(f"Error applying blur to background image: {e}")
        
    print(f"Background image: {type(background_image)}: {background_image.size[0]}x{background_image.size[1]}")
    print(f"Mask image: {type(mask)}: {mask.size[0]}x{mask.size[1]}")
    composite_image = Image.composite(foreground_image, background_image, mask)
 
    #
    # Final Render
    #
    final_params = config_loader.get_final_parameters()
    
    # resize composite_image to match final_params width/height
    composite_image = composite_image.resize((final_params.get("width", processed_width), final_params.get("height", processed_height)), Image.LANCZOS)
    if composite_image.mode == "RGBA":
        # Create a white background (or any solid color you prefer)
        tmp_bg = Image.new("RGB", composite_image.size, (255, 255, 255))
        # Composite the RGBA image onto the background; this removes transparency
        composite_image = Image.alpha_composite(tmp_bg.convert("RGBA"), composite_image).convert("RGB")
   
    # Set seed for reproducibility# generate random integer between 0 and max int   
    random_seed = int(time.time() * 1000) % np.iinfo(np.int32).max 
    generator = torch.manual_seed(random_seed)
     
    if FINAL_PIPELINE is not None: 
        print(f"🎭 Final Image to Image")
        result = FINAL_PIPELINE(
            prompt=prompt + "," + bg_prompt,
            width=final_params.get("width", processed_width),
            height=final_params.get("height", processed_height),
            negative_prompt=final_params.get("negative_prompt", ""),
            guidance_scale=final_params.get("guidance_scale", 7.5),
            num_inference_steps=final_params.get("num_inference_steps", 20),
            image=composite_image,
            strength=final_params.get("strength", 0.5),
            generator=generator,
            num_images_per_prompt=4
        )
        final_image1 = result.images[0]
        final_image2 = result.images[1]
        final_image3 = result.images[2]
        final_image4 = result.images[3]
    else:
        final_image1 = composite_image
        final_image2 = composite_image
        final_image3 = composite_image
        final_image4 = composite_image
    
    return [final_image1, final_image2, final_image3, final_image4, composite_image, foreground_image, background_image]