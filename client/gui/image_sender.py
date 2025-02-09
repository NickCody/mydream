import cv2
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PyQt5 import QtCore
import requests

# --- Utility: Retry Session ---
def get_retry_session(
    retries=5,
    backoff_factor=1,
    status_forcelist=(500, 502, 503, 504),
    session=None
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# --- Image Sender (for posting a frame to the server) ---
class ImageSender(QtCore.QThread):
    """
    A QThread subclass that sends a frame to the server for processing.
    Emits finished_signal with the processed image bytes on success or error_signal on failure.
    """
    finished_signal = QtCore.pyqtSignal(bytes)
    error_signal = QtCore.pyqtSignal(str)
    
    def __init__(self, frame, prompt, server_url, parent=None):
        super(ImageSender, self).__init__(parent)
        self.server_url = server_url
        self.frame = frame
        self.prompt = prompt
        
    def run(self):
        try:
            print("ImageSender.run: processing frame...")

            # Encode the frame as PNG with max compression (9)
            ret, png_data = cv2.imencode('.png', self.frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if not ret:
                self.error_signal.emit("Failed to encode frame.")
                return
            
            files = {
                "image": ("frame.png", png_data.tobytes(), "image/png")  # Updated to PNG
            }
            data = {"prompt": self.prompt}
            
            # Send the request using a retry-enabled session.
            session = get_retry_session(retries=5, backoff_factor=1)
            #response = session.post("http://localhost:8000/api/process", files=files, data=data)
            response = session.post(f"{self.server_url}/api/process", files=files, data=data)
            
            if response.status_code == 200:
                self.finished_signal.emit(response.content)
            else:
                self.error_signal.emit(f"Server error: {response.status_code}")
        except Exception as e:
            self.error_signal.emit(str(e))
