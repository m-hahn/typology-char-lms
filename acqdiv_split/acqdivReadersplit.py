from config import ACQDIV_HOME

import multiprocessing
import os
import random
#import accessISWOCData
#import accessTOROTData
import sys

import edlib

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
      basePath = ACQDIV_HOME+"/tsv/acqdiv_final_data/"+language.lower()+"/"
      self.morphemes = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("morphemes") and x.endswith(".tsv")])

      self.speakers = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("speakers") and x.endswith(".tsv")])
      if split == "train":
	      self.utterances = readTSV([basePath+x for x in os.listdir(basePath) if x.startswith("utterances_train") and x.endswith(".tsv")])
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
        yield (sentence[utterances_raw_index]+((" " if blankBeforeEOS else "")+"\n" if markUtteranceBoundaries else "")).lower()


   def iteratorMorph(self, markUtteranceBoundaries=True, blankBeforeEOS=True):
     utterance_raw_index = self.utterances[0].index(self.UTTERANCE_COLNAME)
     morpheme_index = self.utterances[0].index("morpheme")
     gloss_raw_index = self.utterances[0].index("gloss_raw")
     pos_raw_index = self.utterances[0].index("pos_raw")
     counter = 0
     for sentence in self.utterances[1]:
        startTime = time.time()
        counter += 1
        if counter % 500 == 0:
           print((counter/len(self.utterances[1]), counter))
        utterance_raw = (sentence[utterance_raw_index]).lower()
        utterance_for_return = utterance_raw+((" " if blankBeforeEOS else "")+"\n" if markUtteranceBoundaries else "")
        utterance = [x for x in utterance_raw.lower().split(" ") if x != ""]
        morpheme = [x for x in sentence[morpheme_index].split(" ") if x != ""]
        gloss_raw = [x for x in sentence[gloss_raw_index].split(" ") if x != ""]
        pos_raw = [x for x in sentence[pos_raw_index].split(" ") if x != ""]
        if self.language == "Japanese":
          for l in [morpheme, gloss_raw, pos_raw]:
            while len(l) < len(utterance):
              l.append("")
              assert time.time() -startTime < 10
        # japanese: expect same length, or annotation missing altogether
        # sesotho: annotation matching, but utterance may have different tokenization --> best chance seems to be heuristic alignment
        # indonesian: some splitting of words into morphemes, with combined whitespace and "-"


        utteranceSegmentedIntoMorphemes = [[x] for x in utterance]


        printHere= False and (random.random() > 0.99)
        if printHere:
              print(110)
#              quit()

        if self.language == "Sesotho":
#             print(".....")
#             print(utterance)
#             print(morpheme)
#             print(gloss_raw)
#             print(pos_raw)
             if len(morpheme) != len(pos_raw):
                _ = 0
                print("WARNING")
                print(utterance)
                print(morpheme)
                print(pos_raw)
                continue

             elif len("".join(morpheme).replace("-", "").lower()) == 0:
                 _ = 0
                 print("WARNING")
                 print(utterance)
                 print(morpheme)
                 print(pos_raw)
                 continue


             else:
               print(morpheme)
               print(pos_raw)
               # align morphemes with the tags
               for j in range(len(morpheme)):
                   assert len(morpheme[j].split("-")) == len(pos_raw[j].split("-"))
               rawUtterance = "".join(utterance)
               rawMorphemes = "".join(morpheme).replace("-", "").lower()
               assert len(rawMorphemes) > 0, morpheme
               #print(rawUtterance)
               #print(rawMorphemes)
             #  print("Starting to align")
               cigar = []
               queue.put(cigar)
               def foo(queue):
      #           print("Entering foo")
                 cigarResu = edlib.align(rawMorphemes, rawUtterance, task="path")["cigar"]
                 cigarList = queue.get()
                 cigarList.append(cigarResu)
                 queue.put(cigarList)
    #             print(cigar)
     #            print("Exiting foo")
              # if (counter/len(self.utterances[1])) > 0.6277:
              #    print((counter/len(self.utterances[1]), rawMorphemes, rawUtterance))
              #    print(rawMorphemes)
              #    print(rawUtterance, flush=True)

#               p = multiprocessing.Process(target=foo, args=(queue,))
 #              p.start()
               foo(queue)
  #             p = multiprocessing.Process(target=foo, args=(queue,))
   #            p.start()
    #           p.join(2)
     #          if p.is_alive():
      #           print("Killing alignment")
       #          p.terminate()
        #         p.join()
               cigar = queue.get()[0]
#               print(p.exitcode)
 #              print("Joined")
   #            print([163, (cigar)])
  #             assert len(cigar) == 1
   #            cigar = cigar[0]["cigar"] #edlib.align(rawMorphemes, rawUtterance, task="path")["cigar"]
               #print("Ending alognment")
               utteranceSegmentedIntoMorphemes = []
               currentUtterancePortion = ""
               positionInRawUtterance = 0
               positionInMorphemes = 0
               morphemesString = " ".join(morpheme)
              # print(cigar)
               j = 0
               while j < len(cigar):
                  assert time.time() -startTime < 10
                  k = j
                  while k < len(cigar):
                     assert time.time() -startTime < 10
                     if "0" <= cigar[k] and cigar[k] <= "9":
                        k += 1
                     else:
                            break
                  length = int(cigar[j:k])
                  action = cigar[k]
                  j = k+1
                  for k in range(length):
                      assert time.time() -startTime < 10
                  #    print((currentUtterancePortion, utteranceSegmentedIntoMorphemes, "rawUtterance", rawUtterance[positionInRawUtterance:], "morphemeString", morphemesString[positionInMorphemes:]))
                      while positionInMorphemes < len(morphemesString) and morphemesString[positionInMorphemes] in [" ","-"]:
                          assert time.time() -startTime < 10
                          utteranceSegmentedIntoMorphemes.append(currentUtterancePortion)
                          currentUtterancePortion = ""
                          positionInMorphemes+= 1

                      if action in ["X", "="]:
                          assert positionInRawUtterance < len(rawUtterance)
                          currentUtterancePortion += rawUtterance[positionInRawUtterance]
                      elif action == "D":
                          currentUtterancePortion  += rawUtterance[positionInRawUtterance]
#                      elif action == "D":

                      if action != "I":
                        positionInRawUtterance += 1
                      if action != "D":
                          positionInMorphemes +=1

               utteranceSegmentedIntoMorphemes.append(currentUtterancePortion)
          #     print(utteranceSegmentedIntoMorphemes)    # for Sesotho, now make this the tokenization
               utteranceReTokenized = []
               utteranceSegmentedIntoMorphemesFinal = []
               indexSegments = 0
               for word in morpheme:
                    assert time.time() -startTime < 10
                    components = len(word.split("-"))
                    utteranceReTokenized.append("".join(utteranceSegmentedIntoMorphemes[indexSegments:indexSegments+components]))
                    utteranceSegmentedIntoMorphemesFinal.append(utteranceSegmentedIntoMorphemes[indexSegments:indexSegments+components])
                    indexSegments += components
               utteranceSegmentedIntoMorphemes = utteranceSegmentedIntoMorphemesFinal
        #       print(utteranceReTokenized)
         #      print(utteranceSegmentedIntoMorphemes)
               utterance = utteranceReTokenized
               assert len(utterance) == len(morpheme)
#        else:
#                assert pos_raw[0] == "none" and len(pos_raw) == 1
#             else:
#                assert len(morpheme) == len(pos_raw)
#        if random.random() > 0.95:
#            quit()
        if True: #False and  self.language == "Sesotho":
           if len(morpheme) == 0 or "".join(morpheme).replace("-","") == "":
                morpheme = utterance[:]
           if len(gloss_raw) == 1 and gloss_raw[0] == "???":
                gloss_raw = ["???" for _ in utterance]
           if len(pos_raw) == 1 and pos_raw[0] == "none":
                pos_raw = ["none" for _ in utterance]
           if len(utterance) != len(morpheme):
               print("WARNING")
               print((utterance, morpheme, gloss_raw, pos_raw, utteranceSegmentedIntoMorphemes))
               continue

        assert len(utterance) == len(morpheme), (utterance, morpheme, gloss_raw, pos_raw, utteranceSegmentedIntoMorphemes)
        assert len(utterance) == len(gloss_raw), (utterance, morpheme, gloss_raw, pos_raw, utteranceSegmentedIntoMorphemes)
        assert len(utterance) == len(pos_raw), (utterance, morpheme, gloss_raw, pos_raw, id_raw, utteranceSegmentedIntoMorphemes)
        assert len(utterance) == len(utteranceSegmentedIntoMorphemes), (utterance, morpheme, gloss_raw, pos_raw,  utteranceSegmentedIntoMorphemes)
        annotated = list(zip(utterance, morpheme, gloss_raw, pos_raw,  utteranceSegmentedIntoMorphemes))
        print(annotated)
        if printHere:
              print(201)
        utterance_for_return = (" ".join(utterance))+((" " if blankBeforeEOS else "")+"\n" if markUtteranceBoundaries else "")
        yield (utterance_for_return, annotated)



class AcqdivReaderPartition():
    def __init__(self, reader):
        self.corpus = reader
    def reshuffledIterator(self, markUtteranceBoundaries=True,
                           blankBeforeEOS=True, originalIterator=AcqdivReader.iterator):
        print("Obtaining all the data from the iterator")
        results = list(self.iterator(
           markUtteranceBoundaries=markUtteranceBoundaries,
           blankBeforeEOS=blankBeforeEOS,
           originalIterator=originalIterator))
        random.shuffle(results)
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
	

#reader = AcqdivReaderPartition(AcqdivReader("train", "Chintang"), "train").reshuffledIterator()
#print(list(reader))

#dev_data = AcqdivReaderPartition(AcqdivReader("dev", "Japanese"), "dev").reshuffledIterator()
#print(list(dev_data))

#acqdivCorpusReader = AcqdivReader("train", "Chintang")
#print(list(acqdivCorpusReader))

#acqdivCorpusReader = AcqdivReader("train", args.language)
#iterator = acqdivCorpusReader.iterator()

#AcqdivReaderPartition(AcqdivReader("train", "Chintang"))
#print(list(AcqdivReaderPartition))
