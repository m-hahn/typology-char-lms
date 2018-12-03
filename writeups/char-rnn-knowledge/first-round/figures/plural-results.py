
with open("plural-results-wiki-german-nospaces-bptt-910515909.txt", "r") as inFile:
   data = [["model"] + x.split(" ") for x in inFile.read().strip().split("\n")]
with open("plural-results-wiki-autoencoder.txt", "r") as inFile:
   data += [["baseline"] + x.split(" ") for x in inFile.read().strip().split("\n")]

means = []
keys = set([tuple(x[:4]) for x in data])
for key in keys:
   values = [float(x[4]) for x in data if tuple(x[:4]) == tuple(key)]
   meanValue = sum(values)/len(values)
   means.append((key, meanValue))
   print(key, meanValue)

means = dict(means)

points = range(4,100,4)

models = ["model", "baseline"]
curves = set([x[3] for x in keys])
curveYs = {}
for model in models:
  for curve in curves:
     curveY = []
     for point in points:
        key = (model, str(point), "NSE",  curve)
        curveY.append(means[key])
     print(curveY)
     curveYs[(model, curve)] = curveY


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)



for model in models:
  for curve in curves:
   if "distr" in curve:
       continue
   ys = curveYs[(model, curve)]
   plt.plot(points, ys, label=curve, linewidth=4.0)
  plt.xticks(range(0,100,20))
  plt.legend()
  plt.show()
  plt.savefig("plural-"+model+".pdf")
  plt.close()


for model in models:
  for curve in curves:
   if "distr" not in curve:
       continue
   ys = curveYs[(model, curve)]
   plt.plot(points, ys, label=curve, linewidth=4.0)
  plt.xticks(range(0,100,20))
  plt.legend()
  plt.show()
  plt.savefig("plural-distr-"+model+".pdf")
  plt.close()










