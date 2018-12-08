from config import VOCAB_HOME

from acqdivReadersplit import AcqdivReader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
import random

args=parser.parse_args()
print(args)


acqdivCorpusReader = AcqdivReader("train", args.language)

vocabularychar = set()
vocabulary = set()
iterator = acqdivCorpusReader.iterator()
for utterance in iterator:
   utterance = utterance.split(" ; ")
   for word in utterance:
       vocabulary.add(word)
#print(vocabulary)

iterator = acqdivCorpusReader.iterator()
for utterance in iterator:
   utterancenew = utterance.replace(" ; ", " ")
   utterancenew = utterancenew.split(" ")
   for char in utterancenew:
    if char != "\n":
     vocabularychar.add(char)
print(vocabularychar)


with open(VOCAB_HOME+args.language+'-vocab.txt', "w") as outFile:
 for word in vocabulary:
  word=word.strip()
  outFile.write(word + "\n")

with open(VOCAB_HOME+args.language+'-char.txt', "w") as outcharFile:
 for char in vocabularychar:
  outcharFile.write("%s\n" % char)
  
