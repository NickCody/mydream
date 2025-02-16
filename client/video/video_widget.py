import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import video.background_removal as bg_removal

# --- Video Widget (for live video display) ---
class VideoWidget(QtWidgets.QLabel):
    """
    A QLabel-based widget that captures and displays video frames from the webcam,
    maintaining the aspect ratio while fitting into a fixed 640x480 view.
    """
    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        self.setFixedSize(640, 512)
        self.setScaledContents(False)
        self.cap = cv2.VideoCapture(0)
        self.last_frame = None  # Store the last captured frame
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # roughly 30 FPS
        # Initialize background remover (Change to desired method)
        # self.bg_remover = bg_removal.MaskRCNNSegmentation(threshold=0.8)
        self.bg_remover = bg_removal.DeepLabV3Segmentation()
        
    
    @staticmethod    
    def blend_background(frame, color=(255, 0, 0), alpha=0.5):
        """
        Blends only the background of an RGBA image with a given color while keeping the foreground intact.

        Parameters:
        - frame: The input RGBA image (numpy array).
        - color: The (R, G, B) color to blend the background with.
        - alpha: The blend factor for the background (0.0 = full color, 1.0 = original background).

        Returns:
        - The blended RGB image (numpy array).
        """

        if frame.shape[2] == 4:  # Ensure frame has an alpha channel
            # Extract the alpha channel (0 = fully transparent, 255 = fully opaque) and normalize to 0-1
            alpha_channel = frame[:, :, 3] / 255.0  

            # Create a background mask (1 where background, 0 where foreground)
            background_mask = 1.0 - alpha_channel  

            # Convert the color to an array matching the image dimensions
            color_array = np.full((frame.shape[0], frame.shape[1], 3), color, dtype=np.uint8)

            # Extract the original background from the frame
            original_background = frame[:, :, :3]

            # Blend the original background with the new color
            blended_background = (
                original_background * (1 - alpha)  # Retain some of the original background
                + color_array * alpha  # Mix in the new color
            ).astype(np.uint8)

            # Apply the background mask to keep the blended background where needed
            final_background = (
                blended_background * background_mask[:, :, None]  # Apply mask to blended background
                + original_background * alpha_channel[:, :, None]  # Preserve foreground
            ).astype(np.uint8)

            return final_background
        else:
            print("Warning: Expected an RGBA image but got an RGB image instead.")
            return frame  # Return unchanged if no alpha channel is found
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Apply background removal before converting to RGB
            frame = self.bg_remover.process(frame)
            hinted_frame = self.blend_background(frame, color=(0, 0, 0), alpha=0.8)
            # hinted_frame = frame
             
            # Convert processed frame to RGB for Qt display
            frame_rgb = cv2.cvtColor(hinted_frame, cv2.COLOR_BGR2RGB)
            height, width, channels = frame_rgb.shape
            bytes_per_line = channels * width
            q_img = QtGui.QImage(frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            
            self.last_frame = frame
            
    def get_current_frame(self):
        """Return the most recent successfully captured frame."""
        return self.last_frame

    def close(self):
        self.timer.stop()
        self.cap.release()
