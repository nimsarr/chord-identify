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

while start < len(samples):
    segment = samples[:,0][start:start+num_samples]
    start += num_samples

    # this is the line that creates the spectrogram
    frequencies, times, spectrogram = signal.spectrogram(segment, sample_rate)
    spectrograms.append((frequencies, times, spectrogram))

# prints the spectrogram
plt.pcolormesh(spectrograms[50][1], spectrograms[50][0], \
               10*np.log10(spectrograms[50][2]), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
