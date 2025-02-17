import numpy as np
import wave
import os
import tempfile

from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtCore import QUrl

class ChimePlayer:
    def __init__(self, note1_freq=600, note2_freq=1000, duration=0.3, gap=0.1, sample_rate=44100, volume=80):
        """
        Initializes the ChimePlayer by generating a two-note chime ("ding-dong")
        and saving it to a temporary file. The chime consists of:
        
          - A "ding" note (note1) with frequency note1_freq lasting for 'duration' seconds.
          - A silence of length 'gap' seconds.
          - A "dong" note (note2) with frequency note2_freq lasting for 'duration' seconds.
          
        Parameters:
          note1_freq (float): Frequency of the first note (default: 1000 Hz).
          note2_freq (float): Frequency of the second note (default: 600 Hz).
          duration (float): Duration in seconds for each note.
          gap (float): Silence duration in seconds between the two notes.
          sample_rate (int): Audio sample rate (default: 44100 Hz).
          volume (int): Playback volume (0 to 100).
        """
        self.note1_freq = note1_freq
        self.note2_freq = note2_freq
        self.duration = duration
        self.gap = gap
        self.sample_rate = sample_rate
        self.volume = volume

        # Create the chime file once and store its path.
        self.file_path = self._create_temp_chime_file()
        print(f"Temporary chime file created: {self.file_path}")

        # Set up a persistent QSoundEffect instance to play the file.
        self.sound_effect = QSoundEffect()
        self.sound_effect.setSource(QUrl.fromLocalFile(self.file_path))
        self.sound_effect.setVolume(self.volume / 100.0)

    def _create_temp_chime_file(self):
        """
        Generates the two-note "ding-dong" chime as a WAV file and returns its file path.
        """
        # Create a temporary file that persists (delete=False)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file_name = temp_file.name
        temp_file.close()

        # Create time arrays for each note and for the silence between them.
        t_note = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        t_gap = np.linspace(0, self.gap, int(self.sample_rate * self.gap), endpoint=False)

        # Generate the sine wave for each note (scaled to 0.5 for a pleasant volume)
        note1 = 0.5 * np.sin(2 * np.pi * self.note1_freq * t_note)
        note2 = 0.5 * np.sin(2 * np.pi * self.note2_freq * t_note)
        silence = np.zeros_like(t_gap)

        # Concatenate: first note, silence, then second note.
        audio = np.concatenate((note1, silence, note2))

        # Convert to 16-bit PCM format.
        audio_int16 = np.int16(audio * 32767)

        # Write the audio data to the WAV file.
        with wave.open(temp_file_name, 'wb') as wf:
            wf.setnchannels(1)        # Mono
            wf.setsampwidth(2)        # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return temp_file_name

    def stop(self):
        self.sound_effect.stop()  # Ensure any previous playback is stopped.
        
    def play(self):
        self.sound_effect.play()