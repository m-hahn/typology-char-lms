import sys
import os

language = sys.argv[1]


import random
import subprocess
for _ in range(50):
   myID = random.randint(0, 1000000000)
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", "lm-acqdiv-split.py", "--language="+language, "--myID="+str(myID), f"--out-loss-filename=/u/scr/mhahn/acqdiv/search_random_july2019/{language}_{myID}.txt"]
   subprocess.call(command)

