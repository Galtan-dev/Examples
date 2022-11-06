import matplotlib.pylab as plt
import padasip as pa
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from IPython.display import Audio
import IPython

def zs(a):
    """ 1d data z-score """
    a -= a.mean()
    return a / a.std()


# constants
FILENAME = "Frase_de_Neil_Armstrong.wav"
SAMPLERATE = 44100
n = 300 # filter size
D = 200 # signal delay


# open and process source data
fs, data = wavfile.read(FILENAME)
y = data[:,0].copy()
y = y.astype("float64")
y = zs(y) / 10
N = len(y)

# contaminated with noise
q = np.sin(2*np.pi*1000/99*np.arange(N) + 10.1 * np.sin(2*np.pi/110*np.arange(N)))
d = y + q

# prepare data for simulation
x = pa.input_from_history(d, n)[:-D]
d = d[n+D-1:]
y = y[n+D-1:]
q = q[n+D-1:]

# create filter and filter
f = pa.filters.FilterNLMS(n=n, mu=0.01, w="zeros")
yp, e, w = f.run(d, x)

# display sound players for human validation
print("Original record:")
IPython.display.display(Audio(y, rate=SAMPLERATE))

print("Distorted record:")
IPython.display.display(Audio(d, rate=SAMPLERATE))

print("Enhanced record")
IPython.display.display(Audio(e, rate=SAMPLERATE))