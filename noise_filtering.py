from scipy.io import wavfile
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pylab as plt
import padasip as pa

input = []
target = []
t = []

samp_f1, sound_file1 = wavfile.read('file.wav')
samp_f2, sound_file2 = wavfile.read('file1.wav')

for i in range(500000, 1500000):
    input.append(sound_file1[i])
    target.append(sound_file2[i])

IN = np.reshape(input, [len(input), 1], order="F")
TG = np.reshape(target, [len(target), 1], order="F")

for i in range(0, 1000000):
    t.append(i)
e = []
f = pa.filters.FilterGNGD(n=1, mu=0.085, w="random")
y, e, w = f.run(TG, IN)
WE = np.reshape(e, [1000000, 1], order="F")
le = pa.detection.learning_entropy(WE)
plt.plot(le)
plt.show()