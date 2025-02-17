from PIL import Image, ImageOps
from app.model import PIPELINE, FINAL_PIPELINE, config_loader
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

def transform_image(input_image: Image.Image, prompt: str, bg_prompt: str, processed_width: int, processed_height: int, mask: Image.Image = None) -> Image.Image:
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
    # foreground_image = apply_codeformer(codeformer_config, foreground_image)
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
    # Final Render (before Codeformer)
    #
    final_params = config_loader.get_final_parameters()
    
    # resize composite_image to match final_params width/height
    composite_image = composite_image.resize((final_params.get("width", processed_width), final_params.get("height", processed_height)), Image.LANCZOS)
    if composite_image.mode == "RGBA":
        # Create a white background (or any solid color you prefer)
        tmp_bg = Image.new("RGB", composite_image.size, (255, 255, 255))
        # Composite the RGBA image onto the background; this removes transparency
        composite_image = Image.alpha_composite(tmp_bg.convert("RGBA"), composite_image).convert("RGB")
    
    if FINAL_PIPELINE is not None: 
        print(f"🎭 Final inpaint")
        result = FINAL_PIPELINE(
            prompt=prompt + "," + bg_prompt,
            width=final_params.get("width", processed_width),
            height=final_params.get("height", processed_height),
            negative_prompt=final_params.get("negative_prompt", ""),
            guidance_scale=final_params.get("guidance_scale", 7.5),
            num_inference_steps=final_params.get("num_inference_steps", 20),
            image=composite_image,
            strength=final_params.get("strength", 0.5)
        )
        final_image = result.images[0]
    else:
        final_image = composite_image;
    
    codeformer_config = config_loader.config_entry.get("codeformer", {})
    codeformer_image = enhance_faces(
        config_loader.device,
        final_image,
        enabled=codeformer_config.get("enabled", False),
        weight=codeformer_config.get("weight", 0.7),
        upscale=codeformer_config.get("upscale", 1),
        has_aligned=codeformer_config.get("has_aligned", False),
        paste_back=codeformer_config.get("paste_back", True)
    )
 
    # ✅ Print generated image details after CodeFormer processing
    print("\n📸 Output Image Details (After CodeFormer Enhancement):")
    
    # print(f"Final Image Size: {final_image.size[0]}x{final_image.size[1]}")
    # np_image = np.array(final_image)
    # print(f"Image Stats - Min: {np_image.min()}, Max: {np_image.max()}, Mean: {np_image.mean()}")

    return [final_image, codeformer_image, composite_image, foreground_image, background_image]