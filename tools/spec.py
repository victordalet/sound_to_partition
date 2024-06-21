import sys
import wave
import os
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment


class Spec:

    def __init__(self):
        self.file = sys.argv[1]
        self.song = AudioSegment.from_mp3(self.file)
        self.song.export("file.wav", format="wav")
        self.raw = wave.open('file.wav', 'rb')
        try:
            self.display()
        except Exception as e:
            print(f"Error : ${e}")
        os.remove("file.wav")

    def display(self):
        signal = self.raw.readframes(-1)
        signal = np.frombuffer(signal, dtype="int16")
        f_rate = self.raw.getframerate()
        time = np.linspace(
            0,
            len(signal) / f_rate,
            num=len(signal)
        )
        plt.figure(1)
        plt.title("Sound Wave")
        plt.xlabel("Time")
        plt.plot(time, signal)
        plt.show()


if __name__ == '__main__':
    Spec()
