from scipy.io import wavfile
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import csv
import pandas as pd

input = []
target = []

# data opening
samp_f, sound_file = wavfile.read('file.wav')

# high-pass filter aplication
N = 20
fr = 7800
sos = signal.butter(N, fr, "hp", fs=44100, output="sos")
filtered = signal.sosfilt(sos, sound_file)

# new wav file save
sf.write('processed_file.wav', filtered, 44100)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# new sample - smaller
for i in range(1100000, 1200000):
    input.append(sound_file[i])
    target.append(filtered[i])

# ax1.plot(input)
# ax2.plot(sound_file)
# plt.show()


# detection
target = np.reshape(target, [100000, 1], order="F")
input = np.reshape(input, [100000, 1], order="F")
f = pa.filters.FilterGNGD(n=1, mu=0.3, w="zeros")
y, e, w = f.run(target, input)
le = pa.detection.learning_entropy(w, m=50, order=1)
#
# print(target)
print(w)
# print(input)
#
# plotting data to graph
# ax1.plot(w)
# ax2.plot(e)
# ax3.plot(y)
# plt.show()

print(target)
print(input)

# # ukládání do souboru pro aplikaci
# tar_in = []
# for i in range(0, 100000):     # jen pro exponenciální je konec intervalu 700 000
#     tar_in.append(target[i])
# for i in range(0, 100000):
#     tar_in.append(input[i])
# TAR_IN = np.asarray(tar_in)
# TAR_IN = np.reshape(TAR_IN, [100000, 2], order="F")
# np.savetxt("Second_jump.csv", TAR_IN, delimiter=",")
# ax1.plot(target)
# ax2.plot(input)
# plt.show()
