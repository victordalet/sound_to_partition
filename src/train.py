import os
import sys
from typing import List

import librosa
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize

from src.const import NOTE
from src.test import Test


class Train:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data_dir: str = 'dataset'
        self.classes: List[str] = NOTE
        self.max_epochs: int = int(sys.argv[1])

    @staticmethod
    def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128)):
        data = []
        labels = []

        for i, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(class_dir, filename)
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                    data.append(mel_spectrogram)
                    labels.append(i)

        return np.array(data), np.array(labels)

    def create_model(self):
        data, labels = self.load_and_preprocess_data(self.data_dir, self.classes)
        labels = to_categorical(labels, num_classes=len(self.classes))  # Convert labels to one-hot encoding
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=0.2,
                                                                                random_state=42)

        input_shape = self.X_train[0].shape
        input_layer = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(len(self.classes), activation='softmax')(x)
        self.model = Model(input_layer, output_layer)

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.max_epochs, batch_size=32,
                       validation_data=(self.X_test, self.y_test))
        test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(test_accuracy[1])

    def save_model(self):
        self.model.save('weight.h5')


if __name__ == '__main__':
    train = Train()
    train.create_model()
    train.train_model()
    train.save_model()
    test = Test('dataset/mi/316906__jaz_the_man_2__mi.wav')
    test.test()
