from PIL import Image
import cv2
import numpy as np
import sys
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn

def get_device():
    """
    Returns the available GPU device as a torch.device.
    Checks for Apple's MPS first, then CUDA.
    Exits if no compatible GPU is detected.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("❌ ERROR: No GPU detected. This program requires a CUDA or MPS-compatible device.")
        sys.exit(1)
        
class BaseSegmentation:
    def __init__(self, smooth_edges=True):
        self.smooth_edges = smooth_edges

    def apply_mask(self, image, mask):
        """
        Converts a BGR image to RGBA and applies the given mask to the alpha channel.
        
        Parameters:
          image (numpy.ndarray): Input BGR image.
          mask (numpy.ndarray): Single-channel mask (0-255).
        
        Returns:
          numpy.ndarray: RGBA image with the mask in the A channel.
        """
        # Convert OpenCV BGR to BGRA (adds an alpha channel)
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        if self.smooth_edges:
            # Apply Gaussian blur for soft edges
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
        # Set the alpha channel to the mask
        image_rgba[:, :, 3] = mask
        return image_rgba

# -------------------------------
# 1. BackgroundSubtraction
# -------------------------------
class BackgroundSubtraction(BaseSegmentation):
    def __init__(self, smooth_edges=True):
        super().__init__(smooth_edges)
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
    
    def process(self, image):
        fgmask = self.fgbg.apply(image)
        return self.apply_mask(image, fgmask)

# -------------------------------
# 2. GreenScreenRemoval
# -------------------------------
class GreenScreenRemoval(BaseSegmentation):
    def __init__(self, lower_bound=(35, 40, 40), upper_bound=(85, 255, 255), smooth_edges=True):
        super().__init__(smooth_edges)
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
    
    def process(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        # Invert mask to obtain foreground
        mask_fg = cv2.bitwise_not(mask)
        return self.apply_mask(image, mask_fg)

# -------------------------------
# 3. GrabCutSegmentation
# -------------------------------
class GrabCutSegmentation(BaseSegmentation):
    def __init__(self, smooth_edges=True):
        super().__init__(smooth_edges)
    
    def process(self, image, iterations=3):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, width - 20, height - 20)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
        mask_binary = ((mask == 1) | (mask == 3)).astype(np.uint8) * 255
        return self.apply_mask(image, mask_binary)

# -------------------------------
# 4. DeepLabV3Segmentation
# -------------------------------
class DeepLabV3Segmentation(BaseSegmentation):
    def __init__(self, threshold=0.4, image_size=512, smooth_edges=True):
        super().__init__(smooth_edges)
        self.device = get_device()
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.image_size = image_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_size, self.image_size))
        ])
    
    def process(self, image):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)['out'][0]
        probabilities = torch.softmax(output, dim=0).cpu().numpy()
        person_class_index = 15  # "person" class index in DeepLabV3
        if probabilities.shape[0] <= person_class_index:
            print(f"Warning: Person class index {person_class_index} is out of bounds for model output shape {probabilities.shape}")
        mask_prob = probabilities[person_class_index]
        mask_resized = cv2.resize(mask_prob, (image.shape[1], image.shape[0]))
        mask_resized = np.clip(mask_resized, 0, 1)
        mask_binary = (mask_resized > self.threshold).astype(np.uint8) * 255
        return self.apply_mask(image, mask_binary)

# -------------------------------
# 5. MaskRCNNSegmentation
# -------------------------------
class MaskRCNNSegmentation(BaseSegmentation):
    def __init__(self, threshold=0.66, smooth_edges=True):
        super().__init__(smooth_edges)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])
    
    def process(self, image):
        input_tensor = self.transform(image).to(self.device)
        with torch.no_grad():
            predictions = self.model([input_tensor])[0]
        if len(predictions['masks']) == 0:
            # Return original image with full opacity if no detections
            full_alpha = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
            return self.apply_mask(image, full_alpha)
        combined_mask = torch.zeros_like(predictions['masks'][0][0])
        for mask, score in zip(predictions['masks'], predictions['scores']):
            if score > self.threshold:
                combined_mask = torch.logical_or(combined_mask, mask[0] > 0.5)
        combined_mask = combined_mask.cpu().numpy().astype(np.uint8) * 255
        return self.apply_mask(image, combined_mask)

# -------------------------------
# 6. U2NetSegmentation
# -------------------------------
class U2NetSegmentation(BaseSegmentation):
    def __init__(self, smooth_edges=True):
        super().__init__(smooth_edges)
        self.device = get_device()
        self.model = torch.hub.load("xuebinqin/U-2-Net", "u2net", pretrained=True).to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((320, 320))
        ])
    
    def process(self, image):
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)[0]
        saliency_map = output.squeeze().cpu().numpy()
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        saliency_map = (saliency_map * 255).astype(np.uint8)
        saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
        _, mask = cv2.threshold(saliency_map, 128, 255, cv2.THRESH_BINARY)
        return self.apply_mask(image, mask)

# -------------------------------
# 7. MODNetSegmentation
# -------------------------------
# from modnet.models import MODNetModel

# class MODNetSegmentation(BaseSegmentation):
#     def __init__(self, checkpoint_path="path_to_modnet_pretrained.ckpt", input_size=(512, 512), smooth_edges=True):
#         super().__init__(smooth_edges)
#         self.device = get_device()
#         self.input_size = input_size
#         self.model = MODNetModel(backbone_pretrained=False).to(self.device)
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint["modnet"])
#         self.model.eval()
#         self.transform = T.Compose([
#             T.ToTensor(),
#             T.Resize(self.input_size)
#         ])
    
#     def process(self, image):
#         input_tensor = self.transform(image).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             matte = self.model(input_tensor)[0]
#         matte = matte.squeeze().cpu().numpy()
#         matte = (matte - matte.min()) / (matte.max() - matte.min() + 1e-8)
#         matte = (matte * 255).astype(np.uint8)
#         matte = cv2.resize(matte, (image.shape[1], image.shape[0]))
#         binary_matte = (matte > 128).astype(np.uint8) * 255
#         return self.apply_mask(image, binary_matte)