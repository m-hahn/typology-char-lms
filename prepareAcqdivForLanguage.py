from config import ACQDIV_HOME



import os
import random
#import accessISWOCData
#import accessTOROTData
import sys
 

import csv

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

         if language is not None and "language" in header:
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

import sys
language = sys.argv[1]


# extract data for specific language
basePath = ACQDIV_HOME+"/tsv/"
basePathOut = ACQDIV_HOME+"/tsv/"+language.lower()+"/"

import os
if not os.path.exists(basePathOut):
    os.makedirs(basePathOut)



names = ["speakers","morphemes",  "utterances", "words", "uniquespeakers"]


for name in names:
  dataset = readTSV([basePath+name+".tsv"], language=language)
  printTSV(dataset, basePathOut+name+".tsv")


#reader = AcqdivReader(language)


