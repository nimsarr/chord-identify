import os
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fftshift
from pydub import AudioSegment

mp3dir = './mp3_files'
wavdir = './wav_files'
mp3 = '.mp3'
wav = '.wav'

# check if there is a wav file directory, and make one if there isn't
if not os.path.isdir(wavdir):
    os.mkdir(wavdir)

# https://www.geeksforgeeks.org/python-loop-through-files-of-certain-extensions/
# iterate thru the mp3 files that we have and create parallel wav files where still necessary
for mp3file in os.listdir(mp3dir):
    # need to always make sure that we're starting from mp3 files or else errors will happen
    if mp3file.endswith(mp3):
        base_name = mp3file[0 : len(mp3file) - 4]

        # only create a wav file if there isn't one with that name already
        if not os.path.exists(wavdir + '/' + base_name + wav):
            sound = AudioSegment.from_mp3(mp3dir + '/' + mp3file)
            sound.export(wavdir + '/' + base_name + wav, format="wav")
            print("created " + base_name + wav)
    # print a warning when there is a non mp3 file found in the directory - that shouldn't happen in the first place
    else:
        print("warning: non mp3 file found in mp3 directory")

# now work with files in wav directory
# for now this is just working with the first file for the sake of a proof of concept
wavs = os.listdir(wavdir)
sample_rate, samples = wavfile.read(wavdir + '/' + wavs[0])

start = 0
spectrograms = []
num_samples = sample_rate * 5

# iterating thru all of the samples obtained from wav file
while start < len(samples):
    # obtain the next segment of size num_samples, and increment the index accordingly
    segment = samples[:,0][start:start+num_samples]
    start += num_samples

    # create a spectrogram fro the current segment, and append it to the array of spectrograms
    frequencies, times, spectrogram = signal.spectrogram(segment, sample_rate, nfft=1024)
    spectrograms.append(spectrogram)

from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 24
input_shape = (spectrogram[0].shape[0], spectrogram[0].shape[1], 1)

x_train, x_test, y_train, y_test = train_test_split(spectrograms, '''some form of label data''', test_size=0.1, random_state=4100)

print("x train shape before expanding:", x_train.shape)

# return to original dimensions
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        pilayers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
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
model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
