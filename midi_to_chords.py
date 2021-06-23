import sys
from music21 import converter
from music21.note import Note
from music21.chord import Chord
from functools import reduce


def midi_to_chords(midi_file, time_interval, wav_duration):
    # Get a music stream from midi file
    s = converter.parse(midi_file, quantizePost=False)
    # Flatten
    s = s.flat
    # List of dicts with time in seconds for all elements
    sec_map_list = s.secondsMap  # THIS TAKES A WHILE

    sec_list = []
    # Filter for notes
    # Add tuples to list
    # (Note startTime, Note end_time, Note)
    for elementDict in sec_map_list:
        element = elementDict['element']
        if isinstance(element, Note):
            sec_list.append((elementDict['offsetSeconds'], elementDict['endTimeSeconds'], element))

    # Sort in place (default sort by first item in tuple which is start time)
    sec_list.sort()
    # Get final end time
    end_time = reduce(lambda x, y: x if x[1] >= y[1] else y, sec_list)[1]
    print("MIDI duration:", end_time)

    curr_time = 0
    chords = []
    # While curr_time is before end_time of note that is sounding at the end
    # NOTE: temp bugfix, use wav duration instead of midi duration because of
    #       potential discrepancy which causes different numbers of spectrograms
    #       and chords to be generated from the same midi file
    while curr_time + time_interval <= wav_duration:
        # Get all notes sounding at curr_time
        sounding_notes = []
        idx = 0
        note_time = sec_list[idx]
        # Iterate through sec_list from start
        # While startTime is at or before curr_time
        while note_time[0] <= curr_time:
            # Check that end_time is at or after curr_time
            if note_time[1] >= curr_time:
                sounding_notes.append(note_time[2])
            else:
                # If end_time is before curr_time, we've passed this note,
                # so remove from list so we save time later
                del sec_list[idx]
            idx += 1
            note_time = sec_list[idx]

        # Check if there are any notes at curr_time
        # (If there aren't, there's no chord)
        if len(sounding_notes) > 0:
            chord = Chord(sounding_notes).root().pitchClass  # 0-11
        else:
            chord = 12  # non-existent pitch class 12 means no chord
        chords.append(chord)
        curr_time += time_interval

    # print("Num chords: ", len(chords))
    # for chord in chords:
    #     print(chord)

    return chords

# midi_to_chords('midi_files/1d9d16a9da90c090809c153754823c2b.mid', 0.5)

# sys.modules[__name__] = midi_to_chords
