language = "Japanese"


with open("segmentation-predictions/"+language+"-table.txt", "r") as inFile:
   table = [x.strip().split("\t") for x in inFile]
print("Read table")
header = table[0]
header = dict(list(zip(header, range(len(header)))))
table = table[1:]
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


for i in range(maxPosition):
    if predictedPerPosition[i] is None:
      print(i, characterPerPosition[i], predictedPerPosition[i], truePerPosition[i])   
    else:
      print(i, characterPerPosition[i], predictedPerPosition[i], truePerPosition[i])   
     

