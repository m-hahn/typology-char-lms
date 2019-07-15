from config import ACQDIV_HOME

import multiprocessing
import os
import random
#import accessISWOCData
#import accessTOROTData
import sys

#import edlib

import csv

queue = multiprocessing.Queue()


import time

def readTSV(paths, language=None):
   result = []
   header = None
   assert len(paths) < 10
   paths = sorted(paths)
   print(paths)
   for path in paths:
      print(path)
      with open(path, "r") as inFile:
#         data = csv.reader(inFile, delimiter=",", quotechar='"')
         data = [x.split("\t") for x in inFile.read().strip().split("\n")]
 #        headerNew = data[0]
         if header is None:
            headerNew = data[0]
            data = data[1:]
            header = headerNew
         if language is not None:
            languageIndex = header.index("language")
            print(languageIndex)
            for line in data:
              assert len(line) <= len(header), (header, line)
              if len(line) > len(header):
                print((line, header))
              assert languageIndex < len(line), (header, line)
            data = [x for x in data if x[languageIndex] == language]
         result += data
         assert header == headerNew, (header, headerNew)
   return (header, result)



def printTSV(table, path):
   header, data = table
   with open(path, "w") as outFile:
       outFile.write("\t".join(header)+"\n")
       for line in data:
           outFile.write("\t".join(line)+"\n")



class AcqdivReader():
   def __init__(self, split, language):
      basePath = ACQDIV_HOME + language.lower()+"/"
      self.morphemes = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("morphemes") and x.endswith(".tsv")])

      self.speakers = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("speakers") and x.endswith(".tsv")])
      if split == "traindev":
              self.utterances = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("utterances_traindev") and x.endswith(".tsv")])
      if split == "train":
	      self.utterances = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("utterances_train.") and x.endswith("tsv")])
      if split == "dev":
              self.utterances = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("utterances_dev") and x.endswith(".tsv")])
      if split == "test":
	      self.utterances = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("utterances_test") and x.endswith(".tsv")])
      self.words = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("words") and x.endswith(".tsv")])
      self.uniquespeakers = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("uniquespeakers") and x.endswith(".tsv")])

      random.Random(4656).shuffle(self.utterances[1]) 
      self.language = language
      self.UTTERANCE_COLNAME = {"Japanese" : "segmented.utterance", "Chintang" : "segmented.utterance", "Indonesian" : "segmented.utterance", "Sesotho" : "segmented.utterance"}[self.language]
     


   def length(self):
      return len(self.utterances)

   def iterator(self, markUtteranceBoundaries=True, blankBeforeEOS=True):
     utterances_raw_index = self.utterances[0].index(self.UTTERANCE_COLNAME)
     for sentence in self.utterances[1]:
        yield (sentence[utterances_raw_index]+((";" if blankBeforeEOS else "")+"\n" if markUtteranceBoundaries else "")).lower().split(" ")




class AcqdivReaderPartition():
    def __init__(self, reader):
        self.corpus = reader
    def reshuffledIterator(self, markUtteranceBoundaries=True,
                           blankBeforeEOS=True, originalIterator=AcqdivReader.iterator, seed=random.randint(0, 1000000)):
        print("Obtaining all the data from the iterator")
        results = list(self.iterator(
           markUtteranceBoundaries=markUtteranceBoundaries,
           blankBeforeEOS=blankBeforeEOS,
           originalIterator=originalIterator))
        random.Random(seed).shuffle(results)
        for utterance in results:
            yield utterance

    def iterator(self, markUtteranceBoundaries=True,
                 blankBeforeEOS=True, originalIterator=AcqdivReader.iterator):
        iterator = originalIterator(
           self.corpus,
           markUtteranceBoundaries=markUtteranceBoundaries,
           blankBeforeEOS=blankBeforeEOS)
        for x in iterator:
          yield x
	
