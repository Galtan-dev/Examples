from scipy.io import wavfile
import numpy as np
import soundfile as sf

data1 = []
data2 = []

with open("d1.csv", "r", encoding="utf-8") as hodnoty:
    input_matrix = (np.genfromtxt(hodnoty, delimiter=",", skip_header=0))

for i in range(4, 450):
    data1.append(input_matrix[i, 0])
    data2.append(input_matrix[i, 1])

sample_rate_hz = int(1/0.1)

sf.write('recording1.wav', data1, sample_rate_hz)
sf.write('recording2.wav', data2, sample_rate_hz)

fs1, data11 = wavfile.read('recording1.wav')
fs2, data22 = wavfile.read('recording2.wav')

print(data11)
print(data22)




























