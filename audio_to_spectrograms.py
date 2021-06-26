import os
from os import path

from scipy import signal
from scipy.io import wavfile
from midi2audio import FluidSynth

middir = './midi_files'
wavdir = './wav_files'
mid = '.mid'
wav = '.wav'

# Instructions for required library fluidsynth:
# - install fluidsynth
#   - unix: https://github.com/FluidSynth/fluidsynth/wiki/Download
#   - Windows: https://github.com/FluidSynth/fluidsynth/releases (download zip and add bin/ to PATH)
# - put a soundfont file at ~/.fluidsynth/default_sound_font.sf2
#   - https://sites.google.com/site/soundfonts4u/#h.p_biJ8J359lC5W


def midi_to_spectrograms(midi_file, time_interval):
    base_name = path.basename(midi_file).split('.')[0]
    wav_file = base_name + wav
    # Generate temp wav file from midi file
    FluidSynth().midi_to_audio(midi_file, wav_file)
    # Read generated wav file
    sample_rate, samples = wavfile.read(wav_file)

    # Sample rate is samples per second
    # Samples per spectrogram = Sample rate * Seconds per spectrogram
    sample_offset = int(sample_rate * time_interval)
    wav_duration = len(samples) / sample_rate
    print("WAV duration:", wav_duration)

    # iterating through all of the samples obtained from wav file
    sample_idx = 0
    spectrograms = []
    while sample_idx + sample_offset < len(samples):
        # obtain the next segment of size sample_offset, and increment the index accordingly
        segment = samples[:, 0][sample_idx:sample_idx+sample_offset]
        sample_idx += sample_offset

        # create a spectrogram fro the current segment, and append it to the array of spectrograms
        frequencies, times, spectrogram = signal.spectrogram(segment, sample_rate, nfft=1024)
        spectrograms.append(spectrogram)

    # Remove temp wav file
    os.remove(wav_file)

    return spectrograms, wav_duration

# midi_to_spectrograms('midi_files/05f21994c71a5f881e64f45c8d706165.mid', 0.5)

# sys.modules[__name__] = midi_to_spectrograms
