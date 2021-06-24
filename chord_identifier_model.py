from midi_to_chords import midi_to_chords
from audio_to_spectrograms import midi_to_spectrograms

import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

midi_dataset_filepath = '../lmd_aligned/'
# 0 = C, 2 = D, ..., 11 = B, 12 = N
# N means no chord because no notes
chord_labels = ['C', 'C#/D-', 'D', 'D#/E-', 'E', 'F', 'F#/G-', 'G', 'G#/A-', 'A', 'A#/B-', 'B', 'N']
# Time in seconds for each segment of audio to test
time_interval = 0.5


def generate_data():
    max_files = 1000
    start_idx = 200
    idx = 0
    spectrograms = []
    chords = []
    for filepath in glob.iglob(midi_dataset_filepath + '**/*.mid', recursive=True):
        if idx < start_idx:
            idx += 1
            continue
        if idx >= max_files:
            break
        print("---------------")
        print("Processing midi file", str(idx) + ":")
        try:
            file_sgrams, wav_duration = midi_to_spectrograms(filepath, time_interval)
            file_chords = midi_to_chords(filepath, time_interval, wav_duration)
        except Exception as e:
            print("ERROR, SKIPPING:", e, e.__cause__, e.__annotations__)
            continue
        print("Length of sgrams, chords:", len(file_sgrams), len(file_chords))
        spectrograms.extend(file_sgrams)
        chords.extend(file_chords)
        idx += 1

    return np.array(spectrograms), np.array(chords)


def train_nn(spectrograms, chords):
    # Below code adapted from https://keras.io/examples/vision/mnist_convnet/
    # Model / data parameters
    num_classes = 13 # for each possible chord root pitch class, + no chord
    input_shape = (spectrograms[0].shape[0], spectrograms[0].shape[1], 1)

    # Spectrograms are the data, chords are the labels
    sgrams_train, sgrams_test, chords_train, chords_test = train_test_split(spectrograms, chords, test_size=0.1)

    print("sgrams_train shape before expanding:", sgrams_train.shape)

    # return to original dimensions
    sgrams_train = np.expand_dims(sgrams_train, -1)
    sgrams_test = np.expand_dims(sgrams_test, -1)
    print("sgrams_train shape:", sgrams_train.shape)
    print(sgrams_train.shape[0], "train samples")
    print(sgrams_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    chords_train = keras.utils.to_categorical(chords_train, num_classes)
    chords_test = keras.utils.to_categorical(chords_test, num_classes)

    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(sgrams_train, chords_train, batch_size=64, epochs=100, validation_split=0.1)

    score = model.evaluate(sgrams_test, chords_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


spectrograms, chords = generate_data()
np.save('spectrograms1k1.npy', spectrograms)
np.save('chords1k1.npy', chords)
# spectrograms = np.load('spectrograms.npy')
# chords = np.load('chords.npy')
train_nn(spectrograms, chords)
