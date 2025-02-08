import json
import pyaudio
from vosk import Model, KaldiRecognizer
import os

class SpeechRecognizer:
    """
    A class to capture audio from the default microphone and transcribe it using the Vosk speech recognition engine.
    """

    def __init__(self, model_path: str, rate: int = 24000, chunk: int = 8192):
        """
        Initializes the speech recognizer.
        
        Args:
            model_path (str): Path to the Vosk model directory.
            rate (int): Sampling rate for audio capture.
            chunk (int): Number of audio frames per buffer.
        """
        self.rate = rate
        self.chunk = chunk
        # Load the Vosk model (make sure the model is downloaded to the specified path)
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.rate)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
        self.stream.start_stream()

    def get_transcription(self) -> str:
        """
        Reads a chunk of audio from the microphone and returns the transcription.
        
        Returns:
            str: The recognized text from the audio. Returns an empty string if no speech is detected.
        """
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
        except Exception as e:
            print("Error reading audio stream:", e)
            return ""
        
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            return result.get("text", "").strip()

    def close(self):
        """
        Stops and closes the audio stream and terminates the PyAudio instance.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

