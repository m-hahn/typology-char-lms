


#RNN
#Accuracy
#{'none': 0.7174447174447175, 'sehr': 0.4957002457002457, 'sehr_extrem': 0.5024570024570024, 'sehr_extrem_unglaublich': 0.214987714987715}
#Control
#{'none': 0.492014742014742, 'sehr': 0.3022113022113022, 'sehr_extrem': 0.47665847665847666, 'sehr_extrem_unglaublich': 0.23464373464373464}
#By PMI
#{'none': 0.8538083538083538, 'sehr': 0.878992628992629, 'sehr_extrem': 0.6074938574938575, 'sehr_extrem_unglaublich': 0.2457002457002457}



# LSTM
#Accuracy
#{'none': 0.9263803680981595, 'sehr': 0.9625766871165644, 'sehr_extrem': 0.9447852760736196, 'sehr_extrem_unglaublich': 0.9300613496932515}
#Control
#{'none': 0.5588957055214724, 'sehr': 0.24233128834355827, 'sehr_extrem': 0.2392638036809816, 'sehr_extrem_unglaublich': 0.17668711656441718}
#By PMI
#{'none': 0.9447852760736196, 'sehr': 0.9809815950920245, 'sehr_extrem': 0.9717791411042945, 'sehr_extrem_unglaublich': 0.9717791411042945}

accLSTM =[[0.93, 0.56], [0.96, 0.24], [0.94, 0.24], [0.93, 0.18]] 

accNgram = [[0.75, 0.75], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

accRNN = [[0.71, 0.59], [0.49, 0.30], [0.5, 0.47], [0.21, 0.23]]

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
  plt.savefig("german-restricted-prep-"+genderName+".pdf")
  plt.close()


