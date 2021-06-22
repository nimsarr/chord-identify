import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fftshift
from midi2audio import FluidSynth

middir = './midi_files'
wavdir = './wav_files'
mid = '.mid'
wav = '.wav'

def midi_to_spectrograms():
    # check if there is a wav file directory, and make one if there isn't
    if not os.path.isdir(wavdir):
        os.mkdir(wavdir)

    # https://www.geeksforgeeks.org/python-loop-through-files-of-certain-extensions/
    # iterate thru the mp3 files that we have and create parallel wav files where still necessary
    for midfile in os.listdir(middir):
        # need to always make sure that we're starting from mp3 files or else errors will happen
        if midfile.endswith(mid):
            infile = middir + '/' + midfile
            base_name = midfile[0 : len(midfile) - 4]
            outfile = wavdir + '/' + base_name + wav

            print(infile)
            print(outfile)

            # only create a wav file if there isn't one with that name already
            if not os.path.exists(outfile):
                # sound = AudioSegment.from_mp3(mp3dir + '/' + mp3file)
                # sound.export(wavdir + '/' + base_name + wav, format="wav")
                # TODO THIS LINE IS BROKEN WE NEED TO FIGURE OUT MIDI TO AUDIO
                FluidSynth().midi_to_audio(infile, outfile)
                print("created " + base_name + wav)
        # print a warning when there is a non mp3 file found in the directory - that shouldn't happen in the first place
        else:
            print("warning: non midi file found in mp3 directory")

    # now work with files in wav directory
    # for now this is just working with the first file for the sake of a proof of concept
    wavs = os.listdir(wavdir)
    sample_rate, samples = wavfile.read(wavdir + '/' + wavs[0])

    start = 0
    spectrograms = []
    num_samples = sample_rate # this line means it will be one spectrogram per second

    # iterating thru all of the samples obtained from wav file
    while start < len(samples):
        # obtain the next segment of size num_samples, and increment the index accordingly
        segment = samples[:,0][start:start+num_samples]
        start += num_samples

        # create a spectrogram fro the current segment, and append it to the array of spectrograms
        frequencies, times, spectrogram = signal.spectrogram(segment, sample_rate, nfft=1024)
        spectrograms.append(spectrogram)

    return spectrograms

audio_to_spectrogram()

sys.modules[__name__] = audio_to_spectrogram
