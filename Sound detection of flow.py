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

MAT1 = []
MAT11 = []
MAT2 = []
MAT = []
odečtená_matice = []
# with open("d1.csv", "r", encoding="utf-8") as hodnoty:
#     mat1 = (np.genfromtxt(hodnoty, delimiter=",", skip_header=0))
with open("d2.csv", "r", encoding="utf-8") as hodnoty:
    mat1 = (np.genfromtxt(hodnoty, delimiter=",", skip_header=0))

for i in range(0, 457):
    odečtená_matice.append((mat1[i, 1]-mat1[i, 0]))
tr = np.reshape(odečtená_matice, [457, 1])
print(tr)

for i in range(1, 452):
    MAT1.append(odečtená_matice[i])
for i in range(6, 457):
    MAT11.append(odečtená_matice[i])

input = np.reshape(MAT1, [451, 1], order="F")
output = np.reshape(MAT11, [451, 1], order="F")

f = pa.filters.FilterGNGD(n=1, mu=0.08, w="random")
y, e, w = f.run(output, input)
le = pa.detection.learning_entropy(w)
plt.plot(le)

plt.show()