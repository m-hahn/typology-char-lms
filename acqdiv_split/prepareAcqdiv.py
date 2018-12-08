from config import ACQDIV_HOME



import os
import random
#import accessISWOCData
#import accessTOROTData
import sys
 

import csv


def readCSV(paths):
   result = []
   header = None
   assert len(paths) < 10
   paths = sorted(paths)
   print(paths)
   for path in paths:
      print(path)
      with open(path, "r") as inFile:
         data = csv.reader(inFile, delimiter="#", quotechar='"')
         if header is None:
            headerNew = next(data)
            header = headerNew
         for line in data:
             try:
              assert len(line) == len(header), (line, header)
              result.append(line)
             except:
              pass
         assert header == headerNew, (header, headerNew)
   return (header, result)

def printTSV(table, path):
   header, data = table
   with open(path, "w") as outFile:
       outFile.write("\t".join(header)+"\n")
       for line in data:
           outFile.write("\t".join(line)+"\n")

def mergeCSV(infiles, outfile):
   with open(outfile, "w") as outFile:
      for path in infiles:
          with open(path, "r") as inFile:
             outFile.write(inFile.read())


basePath = ACQDIV_HOME + "/csv/"
basePathOut = ACQDIV_HOME + "/tsv/"
names = ["utterances_train", "utterances_dev", "utterances_test", "speakers", "morphemes", "words", "uniquespeakers"]

for name in names:
  infiles = [basePath+x for x in os.listdir(basePath) if x.startswith(name) and x.endswith(".csv")]
  if len(infiles) > 1:
    mergeCSV(infiles = infiles, outfile = basePath+name+".csv")
  dataset = readCSV([basePath+name+".csv"])
  printTSV(dataset, basePathOut+name+".tsv")

def find_child_lines(speakers_file):
        child_id=[]
        for line_ in speakers_file:
                if "Target_Child" in line_:
                        column=line_.split("\t")
                        if column[0] not in child_id:
                                print(column[0])
                                child_id.append(column[0])
        return child_id


speakers_file=open( (basePathOut + "speakers.tsv"), "r")
list_=(find_child_lines(speakers_file))
print(list_)
speakers_file.close()

utterances_train_file=open((basePathOut + "utterances_train.tsv"), "r")
lines_train=utterances_train_file.readlines()
utterances_train_file.close()

utterances_dev_file=open((basePathOut + "utterances_dev.tsv"), "r")
lines_dev=utterances_dev_file.readlines()
utterances_dev_file.close()

utterances_test_file=open((basePathOut + "utterances_test.tsv"), "r")
lines_test=utterances_test_file.readlines()
utterances_test_file.close()

def remove_child_lines(utterances_file, lines_):
        for line_ in lines_:
                column=line_.split("\t")
                print(column[3])
                if column[3] not in list_:
                        utterances_file.write(line_)
                        

utterances_train_file=open((basePathOut + "utterances_train.tsv"),"w")
remove_child_lines(utterances_train_file, lines_train)
utterances_train_file.close()

utterances_test_file=open((basePathOut + "utterances_test.tsv"),"w")
remove_child_lines(utterances_test_file, lines_test)
utterances_test_file.close()


utterances_dev_file=open((basePathOut + "utterances_dev.tsv"),"w")
remove_child_lines(utterances_dev_file, lines_dev)
utterances_dev_file.close()


