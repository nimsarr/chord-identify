import os
import numpy as np
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
num_samples = sample_rate // 10

# iterating thru all of the samples obtained from wav file
while start < len(samples):
    # obtain the next segment of size num_samples, and increment the index accordingly
    segment = samples[:,0][start:start+num_samples]
    start += num_samples

    # create a spectrogram fro the current segment, and append it to the array of spectrograms
    frequencies, times, spectrogram = signal.spectrogram(segment, sample_rate, nfft=1024)
    spectrograms.append((frequencies, times, spectrogram))

# arbitrary example of creating 2 spectrograms from different parts of the same audio
# proves that we're obtaining real audio with variation in frequency content
for ii in range(0, 2):
    plt.subplot(ii + 211)
    plt.pcolormesh(spectrograms[50 + ii][1], spectrograms[50 + ii][0], \
                        10*np.log10(spectrograms[50 + ii][2]), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

plt.show()
