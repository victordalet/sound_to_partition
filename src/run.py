import os
import sys
from typing import List
from pydub import AudioSegment
from tqdm import tqdm

import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.image import resize
import tensorflow as tf

from src.const import NOTE


class Main:

    def __init__(self):
        self.model = load_model('weight.h5')
        self.target_shape = (128, 128)
        self.classes: List[str] = NOTE
        self.audio_file_path: str = sys.argv[1]
        self.output_file: str = sys.argv[2]
        self.run()

    def test_audio(self, file_path, model):
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), self.target_shape)
        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + self.target_shape + (1,))

        predictions = model.predict(mel_spectrogram)

        class_probabilities = predictions[0]

        predicted_class_index = np.argmax(class_probabilities)

        return class_probabilities, predicted_class_index

    def predict_note(self, name_cut_file: str) -> str:
        class_probabilities, predicted_class_index = self.test_audio(name_cut_file, self.model)

        predicted_class = self.classes[predicted_class_index]
        return predicted_class

    @staticmethod
    def remove_file_in_cut_directory():
        files = os.listdir("cut")
        for file in files:
            os.remove(f"cut/{file}")

    def test_file(self) -> List[str]:
        response: List[str] = []
        files = os.listdir("cut")
        for index in tqdm(range(len(files))):
            response.append(self.predict_note(f"cut/{index}.wav"))
        return response

    def note_to_csv(self, notes: List[str]):
        print(notes)
        f = open(self.audio_file_path, "a")
        for note in notes:
            f.write(f"{note},")
        f.close()

    def separate_song_in_multiple_file(self):
        song = AudioSegment.from_mp3(self.audio_file_path)
        limiter: int = 100
        number_limiter_in_song: int = len(song) // limiter
        for index in range(number_limiter_in_song):
            song[(index * limiter): (index * (2 * limiter))].export(f"cut/{index}.wav", format="wav")

    def run(self):
        self.remove_file_in_cut_directory()
        self.separate_song_in_multiple_file()
        self.note_to_csv(self.test_file())
        self.remove_file_in_cut_directory()


if __name__ == '__main__':
    Main()
