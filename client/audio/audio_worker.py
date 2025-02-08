from PyQt5 import QtCore, QtGui, QtWidgets


# --- Audio Worker (for speech recognition) ---
# This worker uses the SpeechRecognizer class (assumed to be defined in client/audio/speech_recognition.py)
# It continuously captures audio and emits transcriptions (with confidence) to update the prompt textbox.
try:
    from client.audio.speech_recognition import SpeechRecognizer
except ImportError:
    # Dummy fallback if SpeechRecognizer is unavailable.
    class SpeechRecognizer:
        def __init__(self, model_path, rate=16000, chunk=4096):
            pass
        def get_transcription(self):
            return "", 0.0
        def close(self):
            pass

class AudioWorker(QtCore.QThread):
    transcription_signal = QtCore.pyqtSignal(str, float)
    
    def __init__(self, model_path, parent=None):
        super(AudioWorker, self).__init__(parent)
        self.model_path = model_path
        self.running = True
        self.recognizer = SpeechRecognizer(model_path)
    
    def run(self):
        while self.running:
            text = self.recognizer.get_transcription()
            if text:
                self.transcription_signal.emit(text, 0.5)
            self.msleep(100)
        self.recognizer.close()
    
    def stop(self):
        self.running = False
