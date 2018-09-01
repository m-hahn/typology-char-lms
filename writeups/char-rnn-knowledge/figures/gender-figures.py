
accLSTM = [[[0.7, 0.21, 0.09], [0.06, 0.9, 0.03], [0.2, 0.21, 0.59]],
           [[0.63, 0.2, 0.17], [0.15, 0.78, 0.07], [0.31, 0.21, 0.49]],
           [[0.67, 0.19, 0.14], [0.16, 0.76, 0.07], [0.31, 0.22, 0.48]]]

accNgram = [[[0.85, 0.08, 0.07], [0.39, 0.57, 0.04], [0.12, 0.08, 0.8]],
            [[0.34, 0.32, 0.34], [0.32, 0.34, 0.34], [0.33, 0.33, 0.34]],
            [[0.33, 0.33, 0.34], [0.34, 0.34, 0.33], [0.32, 0.34, 0.34]]]

accRNN = [[[0.38, 0.47, 0.15], [0.08, 0.88, 0.05], [0.22, 0.45, 0.33]],
          [[0.2, 0.64, 0.15], [0.11, 0.79, 0.09], [0.19, 0.65, 0.16]],
          [[0.23, 0.54, 0.23], [0.19, 0.55, 0.26], [0.21, 0.55, 0.24]]]

accWord = [[[0.77, 0.15, 0.08], [0.05, 0.92, 0.03], [0.11, 0.14, 0.74]],
           [[0.74, 0.11, 0.14], [0.06, 0.88, 0.06], [0.14, 0.12, 0.74]],
           [[0.83, 0.07, 0.1], [0.09, 0.87, 0.04], [0.18, 0.11, 0.71]]]

names = ["CNLM", "N-Grams", "RNN CNLM", "Word LSTM"]
accs = [accLSTM, accNgram, accRNN, accWord]


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 


for gender, genderName in zip(range(3), ["m", "f", "n"]):
  for name, acc in zip(names, accs):
   plt.plot(range(0, 3), [x[gender][gender] for x in acc], label=name, linewidth=4.0)
  plt.xticks(range(0,3))
  plt.legend()

  plt.show()
  plt.savefig("german-gender-"+genderName+".pdf", bbox_inches='tight')
  plt.close()

for name, acc in zip(names, accs):
 plt.plot(range(0, 3), [sum([x[gender][gender] for gender in range(3)])/3 for x in acc], label=name, linewidth=4.0)
plt.xticks(range(0,3))
plt.legend()

plt.show()
plt.savefig("german-gender-total.pdf", bbox_inches='tight')
plt.close()
  
