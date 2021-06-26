import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# 0 = C, 2 = D, ..., 11 = B, 12 = N
# N means no chord because no notes
chord_labels = ['C', 'C#/D-', 'D', 'D#/E-', 'E', 'F', 'F#/G-', 'G', 'G#/A-', 'A', 'A#/B-', 'B', 'N']
epochs = 50


# Use the data generated and saved to the npy files in this directory
# Train a convolutional neural network for 50 epochs on this data and print the test results
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
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(sgrams_train, chords_train, batch_size=16, epochs=epochs, validation_split=0.1)

    score = model.evaluate(sgrams_test, chords_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


spectrograms = np.load('spectrograms.npy')
chords = np.load('chords.npy')
train_nn(spectrograms, chords)
