from config import TRAJECTORIES_HOME, VISUALIZATIONS_HOME


import os
basePath = TRAJECTORIES_HOME

groups = {"words" : ["undersegmented", "over", "under", "missegmented", "agreement"], "boundaries" : ["boundary_precision", "boundary_recall", "boundary_accuracy"], "tokens" : ["token_precision", "token_recall"], "lexical" : ["lexical_recall", "lexical_precision"]}

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

for language in ["Japanese", "Sesotho", "Indonesian"]:
   prefix = "lm-acqdiv-segmentation-analyses-morph.py_acqdiv-"+language.lower()+"-optimized_EPOCH_"
   files = [x for x in os.listdir(basePath) if x.startswith(prefix)]
   print(basePath+"/"+prefix)
   files = zip(files, range(len(files))) #(x, int(x[len(prefix):])) for x in files]
   files = sorted(files, key=lambda x:x[1])
   data = {}
   for filename, number in files:
       with open(basePath+"/"+filename, "r") as inFile:
           dataNew = [x.split("\t") for x in inFile.read().strip().split("\n")]
           for line in dataNew:
              if line[0] not in data:
                 data[line[0]] = [None for _ in files]
              data[line[0]][number] = float(line[1])
   print(data)
   for group, names in groups.items():
    for name in names:
      datapoints = data[name]
      plt.plot(range(len(datapoints)), datapoints, label=name)
    plt.legend()
    plt.show()
    plt.savefig(VISUALIZATIONS_HOME+language+"_"+group+".png")
    plt.close()




