
accLSTM = [[0.93, 0.58], [0.94, 0.32], [0.94, 0.26], [0.93, 0.2]]

accNgram = [[0.75, 0.75], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

accRNN = [[0.6, 0.44], [0.49, 0.34], [0.4, 0.3], [0.27, 0.28]]

accWord = [[0.91, 0.73], [0.94, 0.73], [0.9, 0.68], [0.9, 0.65]]



names = ["CNLM", "N-Grams", "RNN CNLM", "Word LSTM"]
accs = [accLSTM, accNgram, accRNN, accWord]


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

for gender, genderName in zip(range(2), ["Accuracy", "Control"]):
  for name, acc in zip(names, accs):
   plt.plot(range(0, 4), [x[gender] for x in acc], label=name, linewidth=4.0)
  plt.xticks(range(0,4))
  plt.legend()
  plt.ylim((0.2, 1.0))
  plt.show()
  plt.savefig("german-prep-"+genderName+".pdf")
  plt.close()


