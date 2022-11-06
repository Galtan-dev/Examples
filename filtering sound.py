import sys
from pathlib import Path
import statistics
import os
import matplotlib.pyplot
from PyQt5.QtWidgets import (QMainWindow, QComboBox, QPushButton,\
                             QAction, QLabel, QFileDialog, QTableWidgetItem,\
                             QLineEdit, QTableWidget, QCheckBox, QMessageBox,\
                             QApplication)
from PyQt5.QtGui import (QIcon)
from PyQt5.QtCore import QDate, QTime, QDateTime, Qt
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import numpy as np
import matplotlib.pyplot as plt
import padasip as pa
# Import the os module
import csv
import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import matplotlib.pylab as plt
import padasip as pa
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from IPython.display import Audio
import IPython



IN_matr = []
OUT_matr = []
SAMPLERATE = 44100

# with open("d1.csv", "r", encoding="utf-8") as hodnoty:
#     input_matrix = (np.genfromtxt(hodnoty, delimiter=",", skip_header=0))
with open("d2.csv", "r", encoding="utf-8") as hodnoty:
    input_matrix = (np.genfromtxt(hodnoty, delimiter=",", skip_header=0))

for i in range(4, 450):
    IN_matr.append(input_matrix[i, 0])
    OUT_matr.append(input_matrix[i, 1])
d = np.asarray(IN_matr)
d = np.reshape(d, [446, 1])
x = np.asarray(OUT_matr)
x = np.reshape(x, [446, 1])
print(d.shape)
print(x.shape)

try:
    f = pa.filters.FilterGNGD(n=1, mu=0.01, w="zeros")
    y, e, w = f.run(d, x)
except Exception as ex:
    print(ex)

le = pa.detection.learning_entropy(w)


plt.plot(le)
plt.show()











