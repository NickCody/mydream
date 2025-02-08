import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
import traceback
import numpy as np
from app.transform import transform_image  # Your img2img/inpainting pipeline

app = FastAPI()

# Global counter for saved images
image_counter = 1

@app.post("/api/process")
async def process_image(
    image: UploadFile = File(...),  # Expect an image file upload
    prompt: str = Form(...)         # Expect a text prompt as form data
):
    """
    Endpoint to process an input image with a given prompt.
    
    Expects:
      - image: The input image file (supports PNG with transparency).
      - prompt: A text prompt describing the transformation.
    
    Returns:
      - A PNG image where the AI has inpainted the missing background.
    """
    
    print("process_image")
    try:
        # STEP 1: Read the uploaded image bytes
        image_bytes = await image.read()

        # STEP 2: Open the image with PIL
        input_image = Image.open(io.BytesIO(image_bytes))

        # STEP 3: Handle transparency (Extract Alpha Channel)
        if input_image.mode == "RGBA":
            print("Detected transparent image, extracting alpha mask.")
            
            # Extract RGB and Alpha (transparency) mask
            alpha = input_image.getchannel("A")  
            
            # Create a binary mask (white = keep, black = generate)
            mask = Image.new("L", input_image.size, 0)
            mask.paste(alpha, (0, 0))  # Apply alpha as the inpainting mask
            
            # Convert mask to NumPy array for debugging (optional)
            mask_np = np.array(mask)
            mask.save("temp/mask.png")  # Save mask for debugging (optional)
            
            # Remove transparency from original image (so model sees only the subject)
            input_image = input_image.convert("RGB")

        else:
            # Ensure the image is in RGB mode (standard for diffusion models)
            input_image = input_image.convert("RGB")
            mask = None  # No mask needed for non-transparent images

        # STEP 4: Process the image using AI inpainting
        output_image = transform_image(input_image, prompt, mask=mask).convert("RGB")

        # STEP 5: Save the output image as PNG (to preserve transparency)
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)

        # STEP 6: Save the output image to disk for debugging
        global image_counter
        filename = f"temp/rumple{image_counter:02d}.png"
        output_image.save(filename, format="PNG")
        print(f"Saved processed image as {filename}")
        image_counter += 1
        
        # STEP 7: Return the processed image
        return Response(content=buffer.getvalue(), media_type="image/png")

    except Exception as e:
        # Handle errors and return a failure response
        error_details = traceback.format_exc()
        print("Error processing image:", error_details)
        raise HTTPException(status_code=500, detail=str(e))