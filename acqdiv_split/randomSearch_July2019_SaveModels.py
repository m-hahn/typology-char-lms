import sys
import os

language = sys.argv[1]


import random
import subprocess
for _ in range(50):
   myID = random.randint(0, 1000000000)
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", "lm-acqdiv-split.py", "--language="+language, "--myID="+str(myID), f"--out-loss-filename=/u/scr/mhahn/acqdiv/search_random_july2019/{language}_{myID}.txt", f"--save-to={language}_search_{myID}"]
#   command += "--batchSize=32 --char_embedding_size=50 --hidden_dim=512 --layer_num=1 --learning_rate=10 --weight_dropout_hidden=0.5 --weight_dropout_in=0.05".split(" ")
   subprocess.call(command)

