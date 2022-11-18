from scipy.io import wavfile
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pylab as plt
import padasip as pa

input = []
target = []

# data opening
samp_f, sound_file = wavfile.read('file.wav')

# high-pass filter aplication
N = 20
fr = 7000
sos = signal.butter(N, fr, "hp", fs=44100, output="sos")
filtered = signal.sosfilt(sos, sound_file)

# new wav file save
sf.write('processed_file.wav', filtered, 44100)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# new sample - smaller
for i in range(500000,1500000):
    input.append(sound_file[i])
    target.append(filtered[i])
ax1.plot(input)
ax2.plot(target)
sf.write('processed_sound_file.wav', input, 44100)

# filtering and detection
# convertion to matrix
target = np.reshape(target, [1000000, 1], order="F")
input = np.reshape(input, [1000000, 1], order="F")

# filtering
f = pa.filters.FilterGNGD(n=1, mu=0.11, w="random")
y, e, w = f.run(target, input)

# detection
le = pa.detection.learning_entropy(w)

# plotting data to graph
ax3.plot(le)
plt.show()
