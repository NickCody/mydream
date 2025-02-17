from app.transform import transform_image  # Your img2img/inpainting pipeline
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from PIL import Image
import base64
import io
import os

app = FastAPI()

# Global counter for saved images
image_counter = 1
img_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")

def save_image(moniker, image):
    """
    Saves a PIL image to disk with a given moniker. The filename format is:
      temp/rumple_<img_prefix>_<moniker>_<image_counter>.png

    For example:
      rumple_20250216-210957_final_10.png
    """
    global image_counter, img_prefix
    
    # mkdir temp if not exists
    if not os.path.exists("temp"):
        os.makedirs("temp")
        
    filename = f"temp/rumple_{img_prefix}_{image_counter:d}_{moniker}_.png"
    image.save(filename, format="PNG")
    print(f"Saved processed image as {filename}")
    

@app.post("/api/process")
async def process_image(
    image: UploadFile = File(...),  # Expect an image file upload
    prompt: str = Form(...),  
    bg_prompt: str = Form(...),
    processed_width: int = Form(...),
    processed_height: int = Form(...)
):
    """
    Process an input image and return two images:
      - The original RGBA image (with transparency)
      - The final inpainted output image
    """
    try:
        # STEP 1: Read the uploaded image bytes
        image_bytes = await image.read()

        # STEP 2: Open the original image with PIL (keeping its original mode)
        original_image = Image.open(io.BytesIO(image_bytes))
        print(f"🔹 original_image: {original_image.mode}, size: {original_image.size}")

        # STEP 3: Handle transparency for inpainting (create mask)
        if original_image.mode == "RGBA":
            print("Detected transparent image, extracting alpha mask.")
            # Extract the alpha channel for mask
            alpha = original_image.getchannel("A")
            mask = Image.new("L", original_image.size, 0)
            mask.paste(alpha, (0, 0))
            # Remove transparency for processing
            input_image = original_image.convert("RGB")
        else:
            print("Error: Non-transparent image detected, skipping!")
            return JSONResponse(status_code=400, content={"error": "Non-transparent image provided."})

        # STEP 4: Process the image using AI inpainting
        # Note: transform_image returns several images. Here we assume output_img is the final image.
        [output_img, codeformer_img, composite_img, foreground_img, background_img] = transform_image(
            input_image, prompt, bg_prompt, processed_width, processed_height, mask=mask
        )

        # Optionally, save intermediate images for debugging
        save_image("A-input_img", input_image)
        save_image("B-foreground_img", foreground_img)
        save_image("C-background_img", background_img)
        save_image("D-composite", composite_img)
        save_image("E-final", output_img)
        save_image("F-codeformer", codeformer_img)

        global image_counter
        image_counter += 1

        # STEP 5: Convert final output image to PNG bytes
        buffer_final = io.BytesIO()
        output_img.save(buffer_final, format="PNG")
        buffer_final.seek(0)
        final_bytes = buffer_final.getvalue()

        # STEP 6: Convert original image (with transparency) to PNG bytes
        buffer_orig = io.BytesIO()
        input_image.save(buffer_orig, format="PNG")
        buffer_orig.seek(0)
        orig_bytes = buffer_orig.getvalue()

        # STEP 7: Return both images in a JSON response (base64-encoded)
        return JSONResponse(content={
            "original": base64.b64encode(orig_bytes).decode('utf-8'),
            "final": base64.b64encode(final_bytes).decode('utf-8')
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})