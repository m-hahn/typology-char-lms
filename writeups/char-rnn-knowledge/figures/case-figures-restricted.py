
accLSTM = [[0.9289099526066351, 0.9372037914691943], [0.9715639810426541, 0.8021327014218009], [0.9703791469194313, 0.8933649289099526], [0.9917061611374408, 0.6943127962085308]]

accNgram = [[0.95, 0.97], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

accRNN = [[0.9549763033175356, 0.6860189573459715], [0.8696682464454977, 0.48933649289099523], [0.8104265402843602, 0.2843601895734597], [1.0, 0.001184834123222749]]

accWord = [[0.97, 0.98], [0.98, 0.94], [1.0, 0.75], [1.0, 0.71]]



names = ["CNLM", "N-Grams", "RNN CNLM", "Word LSTM"]
accs = [accLSTM, accNgram, accRNN, accWord]


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 



for gender, genderName in zip(range(2), ["Dative", "Genitive"]):
  for name, acc in zip(names, accs):
   plt.plot(range(0, 4), [x[gender] for x in acc], label=name, linewidth=4.0)
  plt.xticks(range(0,4))
  plt.legend()
  plt.show()
  plt.savefig("german-restricted-case-"+genderName+".pdf")
  plt.close()


for name, acc in zip(names, accs):
 plt.plot(range(0, 4), [sum(x)/2 for x in acc], label=name, linewidth=4.0)
plt.xticks(range(0,4))
plt.legend()
plt.show()
plt.savefig("german-restricted-case-total.pdf")
plt.close()

