
names = ["LSTM", "N-Grams", "RNN", "WordNLM", "N-Grams"]
colors = ["blue", "green", "red", "orange"]

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


fig = plt.figure()
fig_legend = plt.figure(figsize=(5, 0.5))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), range(2), range(2), range(2), range(2), range(2), range(2), linewidth=8.0 )
fig_legend.legend(lines, names, loc='center', frameon=False, ncol=4, prop={"size" : 40})
plt.show()
plt.savefig("german-legend.pdf", bbox_inches='tight')
plt.close()


