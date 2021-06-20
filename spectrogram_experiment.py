import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fftshift
from pydub import AudioSegment

# This code contains a very simple example for how we want to eventually prepare the
# training data for our NN

mp3dir = './mp3_files'
wavdir = './wav_files'

mp3 = '.mp3'
wav = '.wav'

# https://www.geeksforgeeks.org/python-loop-through-files-of-certain-extensions/
for mp3file in os.listdir(mp3dir):
    if mp3file.endswith(mp3):
        base_name = mp3file[0 : len(mp3file) - 4]
        if not os.path.exists(base_name + wav):
            sound = AudioSegment.from_mp3(mp3dir + '/' + mp3file)
            sound.export(wavdir + '/' + base_name + wav, format="wav")
            print("created " + base_name + wav)
    else:
        print("warning: non mp3 file found in mp3 directory")


wavs = os.listdir(wavdir)

print(wavdir + '/' + wavs[0])
sample_rate, samples = wavfile.read(wavdir + '/' + wavs[0])

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
