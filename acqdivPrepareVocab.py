from config import VOCAB_HOME

from acqdivReader import AcqdivReader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
import random

args=parser.parse_args()
print(args)




acqdivCorpusReader = AcqdivReader(args.language)

vocabulary = set()
iterator = acqdivCorpusReader.iterator()
for utterance in iterator:
   utterance = utterance.split(" ")
   for word in utterance:
       vocabulary.add(word)
with open(VOCAB_HOME+args.language+'-vocab.txt', "w") as outFile:
   for word in vocabulary:
      print(word, file=outFile)

