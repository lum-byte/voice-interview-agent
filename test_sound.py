import sounddevice as sd
import numpy as np

# To test if playback works as intended
fs = 44100
t = np.linspace(0, 1, fs)
tone = np.sin(2 * np.pi * 440 * t)

sd.play(tone, fs)
sd.wait()
