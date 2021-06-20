import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fftshift
import matplotlib.pyplot as plt

# This code contains a very simple example for how we want to eventually prepare the
# training data for our NN

sample_rate, samples = wavfile.read('karma_police.wav')

start = 0
spectrograms = []
num_samples = sample_rate // 10

# iterating thru all of the samples obtained from wav file
while start < len(samples):
    # obtain the next segment of size num_samples, and increment the index accordingly
    segment = samples[:,0][start:start+num_samples]
    start += num_samples

    # create a spectrogram fro the current segment, and append it to the array of spectrograms
    frequencies, times, spectrogram = signal.spectrogram(segment, sample_rate, nfft=1024)
    spectrograms.append((frequencies, times, spectrogram))

for ii in range(0, 2):
    plt.subplot(ii + 211)
    # test code - prints an arbitrary spectrogram
    plt.pcolormesh(spectrograms[50 + ii][1], spectrograms[50 + ii][0], \
                        10*np.log10(spectrograms[50 + ii][2]), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

plt.show()
