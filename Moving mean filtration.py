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
for i in range(1000000, 1700000):
    input.append(sound_file[i])
    target.append(filtered[i])

# moving exponential filter

# Python program to
# calculate exponential moving averages

# arr = target
# numbers_series = pd.Series(arr)
# moving_averages = round(numbers_series.ewm(
#     alpha=0.5, adjust=False).mean(), 2)
# target = moving_averages.tolist()
#
# arr2 = target
# numbers_series2 = pd.Series(arr2)
# moving_averages2 = round(numbers_series2.ewm(
#     alpha=0.5, adjust=False).mean(), 2)
# input = moving_averages2.tolist()


# Python program to calculate
# simple moving averages using pandas
# arr = target
# window_size = 30
# numbers_series = pd.Series(arr)
# windows = numbers_series.rolling(window_size)
# moving_averages = windows.mean()
# moving_averages_list = moving_averages.tolist()
# target = moving_averages_list[window_size - 1:]
#
# arr2 = input
# window_size = 30
# numbers_series2 = pd.Series(arr2)
# windows2 = numbers_series2.rolling(window_size)
# moving_averages2 = windows2.mean()
# moving_averages_list2 = moving_averages2.tolist()
# input = moving_averages_list2[window_size - 1:]

# # ukládání výstupu do csv
# tar_in = []
# for i in range(0, 697001):     # jen pro exponenciální je konec intervalu 700 000
#     tar_in.append(target[i])
# for i in range(0, 697001):
#     tar_in.append(input[i])
# TAR_IN = np.asarray(tar_in)
# TAR_IN = np.reshape(TAR_IN, [697001, 2])
# np.savetxt("Sound_det_cumufilt.csv", TAR_IN, delimiter=",")


np.savetxt("Sound_det_target.csv", target, delimiter=",")
np.savetxt("Sound_det_input.csv", input, delimiter=",")