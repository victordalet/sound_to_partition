from typing import List

import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.image import resize
import tensorflow as tf

from src.const import NOTE


class Test:

    def __init__(self, audio_file_path: str):
        self.model = load_model('weight.h5')
        self.target_shape = (128, 128)
        self.classes: List[str] = NOTE
        self.audio_file_path: str = audio_file_path

    def test_audio(self, file_path, model):
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), self.target_shape)
        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + self.target_shape + (1,))

        predictions = model.predict(mel_spectrogram)

        class_probabilities = predictions[0]

        predicted_class_index = np.argmax(class_probabilities)

        return class_probabilities, predicted_class_index

    def test(self):
        class_probabilities, predicted_class_index = self.test_audio(self.audio_file_path, self.model)

        for i, class_label in enumerate(self.classes):
            probability = class_probabilities[i]
            print(f'Class: {class_label}, Probability: {probability:.4f}')

        predicted_class = self.classes[predicted_class_index]
        accuracy = class_probabilities[predicted_class_index]
        print(f'The audio is classified as: {predicted_class}')
        print(f'Accuracy: {accuracy:.4f}')
