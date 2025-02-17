from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

class PlaySound:
    def __init__(self, mp3_filename):
        """
        Initializes the PlaySound class by loading the MP3 file.
        
        Parameters:
          mp3_filename (str): The path to the MP3 file to load.
        """
        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(mp3_filename)))
        self.player.setVolume(100)  # Set volume (0-100)

    def play_sound(self):
        """
        Stops any current playback and starts playing the loaded MP3.
        """
        self.player.stop()  # Ensure any previous playback is stopped.
        self.player.play()