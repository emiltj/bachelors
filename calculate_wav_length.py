import pandas as pd
import numpy
import glob
import wave
import contextlib

files = glob.glob('sz/*.wav') + glob.glob('td/*.wav')

durations = []
for i in files:
    fname = i
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        durations.append(duration)

sum(durations)/len(durations)
numpy.std(durations)

