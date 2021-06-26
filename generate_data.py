from midi_to_chords import midi_to_chords
from audio_to_spectrograms import midi_to_spectrograms

import glob
import numpy as np

midi_dataset_filepath = '../lmd_aligned/'
# Time in seconds for each segment of audio to test
time_interval = 0.5
# Index of midi file to start using
start_idx = 0
# Index of midi file at which to stop generating data
max_idx = 100


# Midi files at the specified filepath will be searched recursively and used in the order found
# Total midi files used is max_idx - start_idx
# BEWARE THAT EACH FILE WILL TAKE UP TO A MINUTE OR MORE TO PROCESS
# Data is only saved at the very end to .npy files in this directory
# The spectrograms take up a lot of space, over 1GB per 100 MIDI files
def generate_data():
    idx = 0
    num_skipped = 0
    spectrograms = []
    chords = []
    for filepath in glob.iglob(midi_dataset_filepath + '**/*.mid', recursive=True):
        if idx < start_idx:
            idx += 1
            continue
        if idx >= max_idx:
            break
        print("---------------")
        print("Processing midi file", str(idx) + ":")
        try:
            file_sgrams, wav_duration = midi_to_spectrograms(filepath, time_interval)
            file_chords = midi_to_chords(filepath, time_interval, wav_duration)
        except Exception as e:
            print("ERROR, SKIPPING:", e, e.__cause__)
            num_skipped += 1
            continue
        print("Length of sgrams, chords:", len(file_sgrams), len(file_chords))
        spectrograms.extend(file_sgrams)
        chords.extend(file_chords)
        idx += 1

    print("Num skipped from errors:", num_skipped)
    return np.array(spectrograms), np.array(chords)


spectrograms, chords = generate_data()
np.save('spectrograms.npy', spectrograms)
np.save('chords.npy', chords)
