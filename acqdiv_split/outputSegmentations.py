
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
args=parser.parse_args()
print(args)



with open("segmentation-predictions/"+args.language+"-table.txt", "r") as inFile:
   table = [x.strip().split("\t") for x in inFile]
print("Read table")
header = table[0]
header = dict(list(zip(header, range(len(header)))))
table = table[1:]
tableR = []
for line in table:
   if len(line) == len(header):
      tableR.append(line)
   elif len(line) == 2:
  #    print(0, line)
      tableR[-1] = tableR[-1] + line
      assert len(tableR[-1]) == len(header), tableR[-1]
 #     print(tableR[-1])
   else:
#      print(1, line)
      tableR.append(line + ["\n"])
table = tableR

index_position = header["PositionID"]
maxPosition = max([int(x[index_position]) for x in table])
predictedPerPosition = [None for x in range((maxPosition)+1)]
truePerPosition = [None for x in range((maxPosition)+1)]
characterPerPosition = [None for x in range((maxPosition)+1)]
index_predicted = header["Prediction"]
index_true = header["Boundary"]
index_character = header["Character"]
counter = 0
for line in table:
   counter += 1
   if counter % 10000 == 0:
      print(counter)
   position = int(line[index_position])
   if predictedPerPosition[position] is None:
     predictedPerPosition[position] = line[index_predicted]
     truePerPosition[position] = line[index_true]
     characterPerPosition[position] = line[index_character]

stringPredicted = ""
stringReal = ""
with open("segmentation-predictions/"+args.language+"-predicted.txt", "w") as outPredicted:
  with open("segmentation-predictions/"+args.language+"-real.txt", "w") as outReal:
    for i in range(maxPosition):
      print(i)
      assert predictedPerPosition[i] is not None
      char = characterPerPosition[i]
      if char == "\n":
          print(stringPredicted, file=outPredicted)
          print(stringReal, file=outReal)
          stringPredicted = ""
          stringReal = ""
      else:
          stringPredicted += char
          stringReal += char
          if predictedPerPosition[i] == "1":
              stringPredicted += " "
          if truePerPosition[i] == "1":
              stringReal += " "
#      print(i, characterPerPosition[i], predictedPerPosition[i], truePerPosition[i])   
#    else:
#      print(i, characterPerPosition[i], predictedPerPosition[i], truePerPosition[i])   
#     




