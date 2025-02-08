import cv2
import numpy as np
import mediapipe as mp
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class BackgroundSubtraction:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
    
    def process(self, image):
        fgmask = self.fgbg.apply(image)
        result = cv2.bitwise_and(image, image, mask=fgmask)
        return result

class GrabCutSegmentation:
    def process(self, image):
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

        # Convert image to UMat for OpenCL acceleration
        u_image = cv2.UMat(image)

        # Convert back to NumPy before passing to grabCut
        image_np = cv2.UMat.get(u_image)

        # Perform GrabCut segmentation
        cv2.grabCut(image_np, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Convert mask to binary
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply mask to the original image
        result = image_np * mask2[:, :, np.newaxis]

        return result
    
class MediaPipeSegmentation:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    def process(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(frame_rgb)
        mask = results.segmentation_mask > 0.5
        background = np.zeros_like(image)
        return np.where(mask[:, :, None], image, background)

class DeepLabV3Segmentation:
    def __init__(self, threshold=0.2, image_size=512, smooth_edges=True):
        """
        Initializes the DeepLabV3 segmentation model.
        
        Parameters:
        - threshold (float): Confidence threshold for masking (default 0.5).
        - image_size (int): Input image resolution for the model (default 512).
        - smooth_edges (bool): Whether to smooth the alpha mask (default True).
        """
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Use Mac GPU
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use NVIDIA GPU
        else:
            self.device = torch.device("cpu")  # Use CPU fallback

        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()

        # Store adjustable parameters
        self.threshold = threshold
        self.image_size = image_size
        self.smooth_edges = smooth_edges

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_size, self.image_size))  # Higher resolution for better segmentation
        ])

    def process(self, image):
        """
        Processes an image to extract a person mask using DeepLabV3.

        Parameters:
        - image (np.ndarray): Input BGR image from OpenCV.

        Returns:
        - np.ndarray: Processed RGBA image with transparency applied to background.
        """
        # Convert OpenCV BGR to PIL RGB
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Transform & move to device
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)['out'][0]  # Get segmentation output

        # Convert logits to probability values (softmax across all classes)
        probabilities = torch.softmax(output, dim=0).cpu().numpy()

        # Extract class index for "person"
        person_class_index = 15  # DeepLabV3 assigns "person" to index 15
        if probabilities.shape[0] <= person_class_index:
            print(f"Warning: Person class index {person_class_index} is out of bounds for model output shape {probabilities.shape}")

        # Get probability mask for "person" class
        mask_prob = probabilities[person_class_index]  # Extract the probability for the person class

        # Debug: Print mask statistics
        # print(f"Mask Stats - Min: {mask_prob.min()}, Max: {mask_prob.max()}, Mean: {mask_prob.mean()}")

        # Resize mask back to original image size
        mask_resized = cv2.resize(mask_prob, (image.shape[1], image.shape[0]))

        # Ensure values are properly scaled
        mask_resized = np.clip(mask_resized, 0, 1)  # Keep values between 0 and 1

        # Apply threshold to create a binary mask
        mask_binary = (mask_resized > self.threshold).astype(np.uint8) * 255

        # Convert OpenCV image to RGBA (add alpha channel)
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        if self.smooth_edges:
            # Apply Gaussian blur for soft edges
            mask_binary = cv2.GaussianBlur(mask_binary, (7, 7), 0)

        # Set background transparency
        image_rgba[:, :, 3] = mask_binary

        return image_rgba

class GreenScreenRemoval:
    def __init__(self, lower_bound=(35, 40, 40), upper_bound=(85, 255, 255)):
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
    
    def process(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        return cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

# Example usage in a video workflow
def main():
    cap = cv2.VideoCapture(0)
    method = MediaPipeSegmentation()  # Change to desired method

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = method.process(frame)
        cv2.imshow('Processed Frame', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
