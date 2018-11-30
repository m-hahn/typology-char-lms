from config import VOCAB_HOME

from acqdivReadersplit import AcqdivReader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
import random

args=parser.parse_args()
print(args)


acqdivCorpusReader = AcqdivReader("train", args.language)

vocabulary = set()
iterator = acqdivCorpusReader.iterator()
for utterance in iterator:
   utterance = utterance.split(" ; ")
   for word in utterance:
       vocabulary.add(word)
print(vocabulary)

with open(VOCAB_HOME+args.language+'-vocab.txt', "w") as outFile:
 for word in vocabulary:
  word=word.strip()
  outFile.write(word + "\n")
  
# for word in vocabulary:
   # for x in word:
    # outFile.write(x)    
#with open(VOCAB_HOME+args.language+'-vocab.txt', "w") as outFile:
#  for line in outFile:
#    if line.rstrip():
#      print(line)
