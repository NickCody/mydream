import cv2

class VideoCaptureHandler:
    """
    A class to handle video capture from a webcam using OpenCV.
    """

    def __init__(self, camera_index: int = 0):
        """
        Initializes the video capture.

        Args:
            camera_index (int): Index of the camera to use (default is 0).
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to access the camera.")
        
    def get_frame(self):
        """
        Captures a single frame from the video stream.

        Returns:
            frame (numpy.ndarray): The captured frame in BGR format.
        Raises:
            RuntimeError: If frame capture fails.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame

    def release(self):
        """
        Releases the video capture resource.
        """
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    # Example usage: display the video feed in a window until 'q' is pressed.
    capture_handler = VideoCaptureHandler()
    try:
        while True:
            frame = capture_handler.get_frame()
            cv2.imshow("Video Capture", frame)
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print("Error:", e)
    finally:
        capture_handler.release()
        cv2.destroyAllWindows()