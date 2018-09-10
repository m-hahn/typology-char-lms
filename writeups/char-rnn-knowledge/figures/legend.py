
names = ["CNLM", "N-Grams", "RNN CNLM", "Word LSTM"]
colors = ["blue", "orange", "green", "red"]

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


fig = plt.figure()
fig_legend = plt.figure(figsize=(5, 0.5))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), range(2), range(2), range(2), range(2), range(2), range(2))
fig_legend.legend(lines, names, loc='center', frameon=False, ncol=4)
plt.show()
plt.savefig("german-legend.pdf")
plt.close()


