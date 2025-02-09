import sys
import cv2
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QSystemTrayIcon
from PyQt5.QtGui import QIcon
from video.video_widget import VideoWidget
from gui.image_display_widget import ImageDisplayWidget
from gui.image_sender import ImageSender
from audio.audio_worker import AudioWorker

DEFAULT_PROMPT = r"""Young (pretty) (beautiful) [alexandra daddario|mary elizabeth winstead], dressed in star trek uniform, ultra-realistic portrait, high quality"""
# DEFAULT_PROMPT = r"""Young (handsome) [pedro pascal|antonio banderas], dressed in toga, wavy luxurious hair"""
# DEFAULT_BG_PROMPT = r"""greek baths, roman statues, servants, highly detailed, ultra-realistic portrait, cinematic bokeh, ultra-sharp"""
DEFAULT_BG_PROMPT = r"""blurred starship bridge, lighted panels, buttons, futuristic displays, bokeh blur, cinematic lighting, highly detailed"""
# --- Main Window (combines video feed, image sending, and audio transcription) ---
class MainWindow(QtWidgets.QMainWindow):
    """
    Main window that integrates:
      - A live video feed.
      - A text box for entering (and updating via audio) a prompt.
      - A Record toggle button to start/stop audio recording.
      - A Clear button to clear the prompt.
      - Automatic image posting: frames are continually captured and sent to the server.
        When a processed image is received, it is displayed for 5 seconds.
        On server error, the system waits 2 seconds before retrying.
      - Audio transcription updates the prompt textbox as the user speaks.
    """
    def __init__(self, server_url):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Rumple My Dream")
        self.server_url = server_url  # Store the server URL in an instance variable
        self.resize(1280, 1024)
        self.sender_thread = None
        self.audio_worker = None  # Only controls audio recording.
        self.request_in_progress = False  # Flag to avoid concurrent server requests.
        self.initUI()
        # Start the image sending loop immediately.
        self.process_next_frame()

    def initUI(self):
        
        # Create System Tray Icon
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("client/reasources/mydream.png"))  # Change to your icon file
        self.tray_icon.setVisible(True)
        
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Live video feed area.
        video_layout = QtWidgets.QHBoxLayout()
        video_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.video_widget = VideoWidget(self)
        video_layout.addWidget(self.video_widget)
        main_layout.addLayout(video_layout)
        
        # 🔹 Create a container layout for the controls
        controls_container = QtWidgets.QVBoxLayout()  # Use QVBoxLayout to stack elements vertically

        # 🔹 Create a horizontal layout for the buttons
        controls_layout = QtWidgets.QHBoxLayout()
        self.record_button = QtWidgets.QPushButton("Record", self)
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self.toggle_recording)
        self.record_button.setFixedHeight(33)
        controls_layout.addWidget(self.record_button)


        # 🔹 Voice Input Section
        voice_layout = QtWidgets.QHBoxLayout()
        self.voice_label = QtWidgets.QLabel("Voice")
        voice_layout.addWidget(self.voice_label)

        self.voice_text_box = QtWidgets.QTextEdit(self)
        self.voice_text_box.setPlaceholderText("Voice input will appear here")
        self.voice_text_box.setFixedHeight(80)
        voice_layout.addWidget(self.voice_text_box)

        self.voice_clear_button = QtWidgets.QPushButton("Clear", self)
        self.voice_clear_button.clicked.connect(lambda: self.voice_text_box.clear())
        self.voice_clear_button.setFixedHeight(33)
        voice_layout.addWidget(self.voice_clear_button)

        # 🔹 Static Input Section
        static_layout = QtWidgets.QHBoxLayout()
        self.static_label = QtWidgets.QLabel("Static")
        static_layout.addWidget(self.static_label)

        self.static_text_box = QtWidgets.QTextEdit(self)
        self.static_text_box.setPlaceholderText("Enter static prompt here")
        self.static_text_box.setPlainText(DEFAULT_PROMPT)
        self.static_text_box.setFixedHeight(80)
        static_layout.addWidget(self.static_text_box)

        self.static_clear_button = QtWidgets.QPushButton("Clear", self)
        self.static_clear_button.clicked.connect(lambda: self.static_text_box.clear())
        self.static_clear_button.setFixedHeight(33)
        static_layout.addWidget(self.static_clear_button)

        # 🔹 Add layouts to controls container
        controls_container.addLayout(controls_layout)  # Record button row
        controls_container.addLayout(voice_layout)  # Voice input row
        controls_container.addLayout(static_layout)  # Static input row

        # 🔹 Finally, add the grouped controls container to the main layout
        main_layout.addLayout(controls_container)
        
        # Processed image display.
        processed_layout = QtWidgets.QHBoxLayout()
        processed_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_label = ImageDisplayWidget(self)
        self.processed_label.setFixedHeight(512)
        processed_layout.addWidget(self.processed_label)
        main_layout.addLayout(processed_layout)
       
    def clear_text(self):
        self.text_box.clear()
    
    def toggle_recording(self, checked):
        """
        Toggle audio recording: when on, start the audio worker for transcription;
        when off, stop it. This does not affect the continuous image sending.
        """
        if checked:
            self.record_button.setText("Stop")
            self.start_audio_worker()
        else:
            self.record_button.setText("Record")
            self.stop_audio_worker()
    
    def start_audio_worker(self):
        """Start the audio worker for continuous speech recognition."""
        if self.audio_worker is None:
            model_path = "client/audio/models/vosk-model-small-en-us-0.15"
            self.audio_worker = AudioWorker(model_path)
            self.audio_worker.transcription_signal.connect(self.update_prompt)
            self.audio_worker.start()
    
    def stop_audio_worker(self):
        """Stop the audio worker if it is running."""
        if self.audio_worker is not None:
            self.audio_worker.stop()
            self.audio_worker.wait()
            self.audio_worker = None
    
    def update_prompt(self, text, conf):
        """
        Update the prompt text box with the recognized speech.
        If nothing is heard, the prompt remains blank.
        """
        self.voice_text_box.setPlainText(text)
    
    @staticmethod    
    def resize_and_crop(frame, target_width=640, target_height=512):
        """
        Resizes an image to fit the given target height while keeping the aspect ratio,
        then crops the width to exactly `target_width` (centered).

        Parameters:
        - frame (numpy.ndarray): Input image.
        - target_width (int): Desired output width after cropping (default: 640).
        - target_height (int): Desired output height after resizing (default: 512).

        Returns:
        - Cropped and resized image (numpy.ndarray).
        """

        # Step 1: Get original dimensions
        orig_height, orig_width = frame.shape[:2]

        # Step 2: Compute new width while keeping aspect ratio
        new_width = int((target_height / orig_height) * orig_width)

        # Step 3: Resize the image while keeping aspect ratio
        frame_resized = cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)

        # Step 4: Crop left/right to get exactly target_width
        crop_x_start = max(0, (new_width - target_width) // 2)  # Center cropping
        crop_x_end = crop_x_start + target_width

        # Ensure we don't crop beyond image bounds
        if crop_x_end > new_width:
            crop_x_end = new_width
            crop_x_start = crop_x_end - target_width

        frame_cropped = frame_resized[:, crop_x_start:crop_x_end]

        return frame_cropped    

    def process_next_frame(self):
        """
        Continually capture a frame from OpenCV and send it to the server.
          - If a request is in progress, check again shortly.
          - If no frame is captured, retry after 2 seconds.
          - On successful response, display the image for 5 seconds.
          - On server error, wait 2 seconds before retrying.
        """
        print("process_next_frame called")
        if self.request_in_progress:
            QtCore.QTimer.singleShot(100, self.process_next_frame)
            return
        
        frame = self.video_widget.get_current_frame()
        if frame is None:
            print("Failed to capture frame.")
            QtCore.QTimer.singleShot(100, self.process_next_frame)
            return
        
        # resize frame to 640x512 maintain aspect ratio and crop left/right
        frame = self.resize_and_crop(frame, 1280, 1024)
        
        # Setup prompts
        self.request_in_progress = True
        voice_prompt = f"({self.voice_text_box.toPlainText().strip()}:1.5)"  # May be blank if no transcription.
        if not voice_prompt:
            voice_prompt = ""  # Explicitly use an empty prompt if nothing is heard.
            
        prompt = f"{voice_prompt}, {self.static_text_box.toPlainText().strip()}"
        bg_prompt = DEFAULT_BG_PROMPT
                
        self.sender_thread = ImageSender(frame, prompt, bg_prompt, self.server_url)
        self.sender_thread.finished_signal.connect(self.handle_server_response)
        self.sender_thread.error_signal.connect(self.handle_server_error)
        self.sender_thread.start()
   
    def handle_server_response(self, image_bytes):
        """Display the processed image for 5 seconds, then resume frame processing."""
        print("handle_server_response called")
        # Convert the received bytes to a QImage.
        processed_image = QtGui.QImage.fromData(image_bytes)
        if processed_image.isNull():
            print("Warning: Processed image is null!")
            self.request_in_progress = False
            QtCore.QTimer.singleShot(2000, self.process_next_frame)
            return

        # Instead of manually scaling and setting a pixmap,
        # we delegate the display to our new ImageDisplayWidget control.
        self.processed_label.set_image(processed_image)

        self.request_in_progress = False
        # Display the processed image for 5 seconds before capturing the next frame.
        QtCore.QTimer.singleShot(0, self.process_next_frame) 
    
    def handle_server_error(self, error_msg):
        """
        If a server error occurs, log the error, clear the processed image,
        and wait 2 seconds before retrying.
        """
        print("Error sending image to server:", error_msg)
        self.request_in_progress = False
        QtCore.QTimer.singleShot(2000, self.process_next_frame)
    
    def closeEvent(self, event):
        print("Closing application...")

        # Stop video processing
        self.video_widget.close()

        # Stop the sender thread properly (don't block)
        if self.sender_thread is not None:
            self.sender_thread.quit()  # Request thread exit
            self.sender_thread.wait(100)  # Wait a max of 1s, then continue

        # Stop the audio worker
        self.stop_audio_worker()

        # Accept close event
        event.accept()

# Global variable to store the server URL
SERVER_URL = "http://localhost:8000"  # Default value

def parse_args():
    """ Parses command-line arguments to get the server URL. """
    global SERVER_URL

    parser = argparse.ArgumentParser(description="Start the RumpleMyDream GUI")
    parser.add_argument("--server-url", type=str, default=SERVER_URL, help="Set the server URL (default: http://localhost:8000)")

    args = parser.parse_args()
    SERVER_URL = args.server_url  # Update the global variable
    print(f"🌐 Server URL set to: {SERVER_URL}")
    
def main():
    """ Main function to start the GUI. """
    parse_args()  # Parse command-line arguments

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(SERVER_URL)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()