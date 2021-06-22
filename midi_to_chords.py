import sys
from music21 import converter
from music21.note import Note
from music21.chord import Chord
from functools import reduce

def midi_to_chords(filename):
    # Get stream from midi file
    s = converter.parse(filename, quantizePost=False)

    # Flatten
    s = s.flat

    # List of dicts with time in seconds for all elements
    secMapList = s.secondsMap # THIS TAKES A WHILE
    secList = []

    # Filter for notes
    # Add tuples to list
    # (Note startTime, Note endTime, Note)
    for elementDict in secMapList:
        element = elementDict['element']
        if isinstance(element, Note):
            secList.append((elementDict['offsetSeconds'], elementDict['endTimeSeconds'], element))

    # Sort in place (default sort by first item in tuple which is start time)
    secList.sort()

    # Get final end time
    endTime = reduce(lambda x, y: x if x[1] >= y[1] else y, secList)[1]

    # Time between chord gets
    interval = 0.5

    currTime = 0

    # List of tuples: (chord, time)
    chordTimes = []

    # While currTime is before endTime of note that is sounding at the end
    while currTime <= endTime:
        soundingNotes = []

        idx = 0
        noteTime = secList[idx]
        # Iterate through secList from start
        # While startTime is at or before currTime
        while noteTime[0] <= currTime:
            # Check that endTime is at or after currTime
            if noteTime[1] >= currTime:
                soundingNotes.append(noteTime[2])
            else:
                # If endTime is before currTime, we've passed this note,
                # so remove from list so we save time later
                del secList[idx]
            idx += 1
            noteTime = secList[idx]

        # Check if there are any notes at currTime
        # (If there aren't, there's no chord)
        if len(soundingNotes) > 0:
            chord = Chord(soundingNotes)
            chordTimes.append((chord, currTime))
        currTime += interval

    print("Num chords: ", len(chordTimes))
    for chord, time in chordTimes:
        print(time, ": ", chord.root(), chord.quality)

    return chordTimes

sys.modules[__name__] = midi_to_chords
