
accLSTM = [[0.88, 0.93], [0.96, 0.76], [0.94, 0.89], [0.99, 0.7]]

accNgram = [[0.94, 0.95], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

accRNN = [[0.93, 0.61], [0.82, 0.46], [0.76, 0.33], [1.0, 0.0]]

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
  #plt.legend()
  plt.show()
  plt.savefig("german-case-"+genderName+".pdf")
  plt.close()


for name, acc in zip(names, accs):
 plt.plot(range(0, 4), [sum(x)/2 for x in acc], label=name, linewidth=4.0)
plt.xticks(range(0,4))
#plt.legend()
plt.show()
plt.savefig("german-case-total.pdf")
plt.close()

