import sys
import os

# Add CodeFormer to sys.path if not already present
sys.path.append("CodeFormer")

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils import img2tensor, tensor2img
from basicsr.archs.codeformer_arch import CodeFormer

codeformer_model = None

# Initialize CodeFormer model
def load_codeformer(device="cuda"):  
    global codeformer_model
    print("🔹 Loading CodeFormer model...")
    
    model_path = load_file_from_url(
        url='https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        model_dir='weights',
        progress=True,
        file_name='codeformer.pth'
    )
    
    codeformer_model = CodeFormer(dim_embd=512, codebook_size=1024).to(device)
    codeformer_model.load_state_dict(torch.load(model_path, map_location=device)['params_ema'])
    codeformer_model.eval()
    print("✅ CodeFormer model loaded successfully!")

def enhance_faces(
    device,
    image: Image.Image,
    enabled=True,
    weight=0.7,
    upscale=1,
    has_aligned=False,
    paste_back=True
):
    if not enabled or codeformer_model is None:
        return image

    # Convert image to OpenCV format
    img = np.array(image.convert("RGB"))[:, :, ::-1]
    
    face_helper = FaceRestoreHelper(upscale, face_size=512, device=device)
    face_helper.read_image(img)
    face_helper.get_face_landmarks_5(only_keep_largest=True)  # 🔹 Keep only the largest face
    face_helper.align_warp_face()

    if len(face_helper.cropped_faces) == 0:
        print("⚠️ No face detected, returning original image.")
        return image

    print(f"✅ Processing the largest detected face")

    with torch.no_grad():
        cropped_face = face_helper.cropped_faces[0]  # Only process the largest face
        face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        face_t = face_t.unsqueeze(0).to(device)

        enhanced_face = codeformer_model(face_t, w=weight)[0]
        enhanced_face = tensor2img(enhanced_face, rgb2bgr=True)

        if enhanced_face is not None:
            face_helper.restored_faces = [enhanced_face]  # 🔹 Store only the largest face
        else:
            print("⚠️ Warning: Face enhancement failed, returning original image.")
            return image

    # Ensure affine matrix is correctly set for a single face
    if paste_back and len(face_helper.restored_faces) == len(face_helper.affine_matrices):
        restored_img = face_helper.paste_faces_to_input_image()
    else:
        print("⚠️ Mismatch detected, returning original image")
        restored_img = face_helper.input_img

    return Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))