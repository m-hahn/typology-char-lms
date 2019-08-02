import sys

language = sys.argv[1]

import os

BASE_PATH = "/u/scr/mhahn/acqdiv/search_random_july2019/"

files = [x for x in os.listdir(BASE_PATH) if x.startswith(language+"_")]

data = []
for name in files:
   with open(BASE_PATH+name, "r") as inFile:
      line = next(inFile).strip()
      minimum_loss = float(line[13:line.index(" epoch=")])
      epoch = line[line.index(" epoch=")+7:line.index(" args=Name")]
      args = line[line.index("args=Name"):]
      data.append((minimum_loss, epoch, name, args))
data = sorted(data, key=lambda x:-x[0])
for l in data:
  print("\t".join([str(x) for x in l]))

      



