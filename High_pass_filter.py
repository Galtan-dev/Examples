from scipy.io import wavfile
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import csv

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
for i in range(1000000, 1700000):
    input.append(sound_file[i])
    target.append(filtered[i])
ax1.plot(input)
ax2.plot(target)
sf.write('processed_sound_file.wav', input, 44100)

# filtering and detection
# convertion to matrix
target = np.reshape(target, [700000, 1], order="F")
input = np.reshape(input, [700000, 1], order="F")

# filtering
f = pa.filters.FilterGNGD(n=1, mu=0.11, w="random")
y, e, w = f.run(target, input)

# detection
le = pa.detection.learning_entropy(w)

# plotting data to graph
ax3.plot(le)
plt.show()




# # ukládání výstupu do csv
# tar_in = []
# for i in range(0, 700000):
#     tar_in.append(target[i])
# for i in range(0, 700000):
#     tar_in.append(input[i])
# TAR_IN = np.asarray(tar_in)
# TAR_IN = np.reshape(TAR_IN, [700000, 2])
# np.savetxt("Sound_det_prep.csv", TAR_IN, delimiter=",")




