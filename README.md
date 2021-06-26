# chord-identify

A library for attempting to automatically extract chord data from audio using a convolutional neural network.

Used with the lmd-aligned part of the Lakh MIDI dataset: https://colinraffel.com/projects/lmd/

Requires FluidSynth to be installed (required by python package midi2audio):
- Unix: https://github.com/FluidSynth/fluidsynth/wiki/Download
- Windows: https://github.com/FluidSynth/fluidsynth/releases (download and extract appropriate zip file and add the filepath to the bin/ folder to your PATH environment variable)
- Put a Soundfont file at \~/.fluidsynth/default_sound_font.sf2 (on Windows '\~' means *your* user folder)
  - We used https://sites.google.com/site/soundfonts4u/#h.p_biJ8J359lC5W

Expects the following file placement:

```
<parent>/
<parent>/lmd-aligned/
<parent>/chord-identify/
```

---

Written for the final project of Northeastern's CS4100 Artificial Intelligence course, Summer I 2021.

By: Simon Bass & Andreas Petrides
