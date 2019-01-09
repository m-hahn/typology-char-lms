
accLSTM = [[0.93, 0.58, 0.95], [0.94, 0.32, 0.95], [0.94, 0.26, 0.96], [0.93, 0.2, 0.96]]

accNgram = [[0.76, 0.74, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

accRNN = [[0.6, 0.44, 0.79], [0.49, 0.34, 0.82], [0.4, 0.3, 0.6], [0.27, 0.28, 0.28]]

accWord = [[0.91, 0.73, 0.88], [0.94, 0.73, 0.9], [0.9, 0.68, 0.87], [0.9, 0.65, 0.87]]

counts = 1629

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
 ys = [x[0] for x in acc]
 plt.plot(range(0, 4), ys, label=name, linewidth=4.0, linestyle="-", color=color)
 se = [None for x in acc]
 for i in range(len(acc)):
     accuracy = ys[i]
     se[i] = 2*sqrt(accuracy*(1-accuracy)/(counts))
 print(se)
 plt.errorbar(x=range(0,4), y=ys, yerr=se, color=color, linewidth=4.0)

 ys = [x[1] for x in acc]
 plt.plot(range(0, 4), ys, label=name, linewidth=4.0, linestyle=":", color=color)
 se = [None for x in acc]
 for i in range(len(acc)):
     accuracy = ys[i]
     se[i] = 2*sqrt(accuracy*(1-accuracy)/(counts))
 print(se)
 plt.errorbar(x=range(0,4), y=ys, yerr=se, linewidth=4.0, linestyle=":", color=color)

plt.xticks(range(0,4))
#plt.legend()
plt.ylim((0.2, 1.0))
plt.show()
plt.savefig("german-prep-with-control-se.pdf", bbox_inches='tight')
plt.close()





