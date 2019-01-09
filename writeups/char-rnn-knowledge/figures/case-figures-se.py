
accLSTM = [[0.88, 0.93], [0.96, 0.76], [0.94, 0.89], [0.99, 0.7]]

accNgram = [[0.94, 0.95], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

accRNN = [[0.93, 0.61], [0.82, 0.46], [0.76, 0.33], [1.0, 0.0]]

accWord = [[0.97, 0.98], [0.98, 0.94], [1.0, 0.75], [1.0, 0.71]]

counts = 4509

names = ["CNLM", "N-Grams", "RNN CNLM", "Word LSTM"]
accs = [accLSTM, accNgram, accRNN, accWord]
colors = ["blue", "orange", "green", "red"]

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

from math import sqrt

for name, acc, color in zip(names, accs, colors):
 ys = [sum(x)/2 for x in acc]
 plt.plot(range(0, 4), ys, label=name, linewidth=4.0, color=color)
 se = [None for x in acc]
 for i in range(len(acc)):
     accuracy = sum(acc[i])/2
     se[i] = 2*sqrt(accuracy*(1-accuracy)/(2*counts))
 print(se)
 plt.errorbar(x=range(0,4), y=ys, yerr=se, linewidth=4.0, color=color)
plt.xticks(range(0,4))
plt.show()
plt.savefig("german-case-total-se.pdf", bbox_inches='tight')
plt.close()

