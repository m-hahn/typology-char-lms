# ~/python-py37-mhahn detectBoundariesUnit_FullText_Revised.py --language Japanese --load-from Japanese


from config import VOCAB_HOME, CHECKPOINT_HOME
#CHAR_VOCAB_HOME


import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--gpu", dest="gpu", type=bool)

import random
random.seed(1) # May nonetheless not be reproducible, since the classifier library doesn't seem to allow setting the seed

parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--char_embedding_size", type=int, default=50)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--layer_num", type=int, default=1)
parser.add_argument("--weight_dropout_in", type=float, default=0.01)
parser.add_argument("--weight_dropout_hidden", type=float, default=0.1)
parser.add_argument("--char_dropout_prob", type=float, default=0.33)
parser.add_argument("--char_noise_prob", type = float, default= 0.01)
parser.add_argument("--learning_rate", type = float, default= 0.1)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=40)


args=parser.parse_args()
print(args)

if args.language == "Japanese":
  neuron = [292]
else:
  assert False

# For putting things on the GPU if the --gpu flag is set
def device(x):
    if args.gpu:
        return x.cuda()
    else:
        return x





from acqdivReadersplit import AcqdivReader, AcqdivReaderPartition

#acqdivCorpusReader = AcqdivReader(args.language)
acqdivCorpusReadertrain = AcqdivReader("traindev", args.language)
# in the end, this will be test, but for now let's do traindev to avoid overfitting our research



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

## read the character vocabulary
itos=[]
with open(VOCAB_HOME + args.language + '-char.txt', "r") as inFile:
     for line in inFile:
      line=line.strip()
      itos.append(line)
itos.append("\n")

print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])


import random
import torch
print(torch.__version__)


####################################################################################
# Create the neural network model

# dropout for recurrent weights (from Merity et al 2018)
from weight_drop import WeightDrop

# Create the neural model
rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()
rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
# prepare dropout on the recurrent weights
rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

# the layer transforming the LSTM output into the logits
output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda() 
char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)

# mask that drops entire characters
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}


# Load the model from the checkpoint, if there is one
if args.load_from is not None:
  checkpoint = torch.load(CHECKPOINT_HOME+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable

####################################################################################


def prepareDatasetChunksTest(data, train=True, offset=0):
      numeric = [0]
      boundaries = [None for _ in range(args.sequence_length+1)]
      boundariesAll = [None for _ in range(args.sequence_length+1)]
      uniquePositionID = [0]

      count = 0
      currentWord = ""
      positionID = 0
      print("Prepare chunks")
      if offset < 0:
         numeric += [0 for _ in range(-offset)]
         uniquePositionID += [0 for _ in range(-offset)]
         boundariesAll[:-offset+5] = ["START" for _ in range(-offset+5)]
      for chunk in data:
       print(146, len(chunk))
       for char in chunk:
         print([char, char in stoi])
         if char == ";":
           if count >= offset:
             boundaries[len(numeric)] = currentWord
             boundariesAll[len(numeric)] = currentWord
             currentWord = ""
           continue
         else:
           if boundariesAll[len(numeric)] is None:
               boundariesAll[len(numeric)] = currentWord

         count += 1
         if count >= offset:
           currentWord += char
           uniquePositionID.append(count)
           numeric.append((stoi[char]+3 if char in stoi else 2))
         if len(numeric) > args.sequence_length:
            yield numeric, boundaries, boundariesAll, uniquePositionID
            numeric = [0] + numeric[args.sequence_length:]
            uniquePositionID = [0] + uniquePositionID[args.sequence_length:]

            assert len(boundaries) == args.sequence_length+1
            assert len(numeric) == len(uniquePositionID)
            boundaries = boundaries[int(args.sequence_length/2):] + [None for _ in range(int(args.sequence_length/2))]
            boundariesAll = boundariesAll[int(args.sequence_length/2):] + [None for _ in range(int(args.sequence_length/2))]
            assert len(boundaries) == args.sequence_length+1


def codeToChar(i):
   if i == 2:
     return "%" # OOV
   else:
     return itos[i-3]



def forward(numeric, train=True, printHere=False):
      global labels_sum
      numeric, boundaries, boundariesAll, uniquePositionID = zip(*numeric)

      numeric_selected = numeric
      input_tensor = Variable(torch.LongTensor(numeric_selected).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric_selected).transpose(0,1)[1:].cuda(), requires_grad=False)

      

      embedded = char_embeddings(input_tensor)

      hidden = None
      print(len(embedded))
      for i in range(int(args.sequence_length/2), len(embedded)):
            out, hidden = rnn_drop(embedded[i].unsqueeze(0), hidden)
            for j in range(len(embedded[0])):
                 nextRelevantWord = ([boundaries[j][k] for k in range(i+2, len(boundaries[j])) if boundaries[j][k] is not None]+["END_OF_SEQUENCE"])[0]
                 target = 1 if boundaries[j][i+1] is not None else 0
                 hidden_states.append((hidden[1][:,j,:].flatten()[neuron[0]]).unsqueeze(0).cpu().detach().numpy())
                 boundary_positions.append((j,i))

                 labels.append(target)
                 labels_sum += labels[-1]

                 relevantWords.append(boundariesAll[j][i+1])
                 
                 #relevantNextWords.append(nextRelevantWord)
                 positionIDs.append(int(uniquePositionID[j][i]))
                 charactersAll.append(codeToChar(int(numeric[j][i])))
                 assert boundariesAll[j][i+1] is not None


####################################################################################3

import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split






training_data = list(AcqdivReaderPartition(acqdivCorpusReadertrain).reshuffledIterator(blankBeforeEOS=True, seed=0))



#############################################


fullDataset_perRound_general = [list(prepareDatasetChunksTest(training_data, train=True, offset=ROUND*(int(args.sequence_length/3)))) for ROUND in range(-2,2)]

with open("segmentation-predictions/"+args.language+"-table.txt", "w") as outFileTable:
  print("\t".join(["Round", "Fold", "PositionID", "Character", "Prediction", "Boundary"]), file=outFileTable)
  for FOLD in range(2):
    print("Got data")
    if FOLD == 0:
       fullDataset_perRound = fullDataset_perRound_general[::]
    elif FOLD == 1:
       fullDataset_perRound = [x[128:200] + x[:128] + x[200:] for x in fullDataset_perRound_general]
    else:
       assert False
    assert args.batchSize == 16
    training_chars = (x for x in fullDataset_perRound[0][:64])
    
    test_chars_perRound = [x[66:] for x in fullDataset_perRound]
    print([len(list(x)) for x in fullDataset_perRound_general])

    wordsSoFar = set()
    hidden_states = []
    labels = []
    relevantWords = []
    #relevantNextWords = []
    labels_sum = 0
    
    boundary_positions = []
    positionIDs = []
    
    charactersAll = []
    
    

    rnn_drop.train(False)
    startTime = time.time()
    trainChars = 0
    counter = 0
    while True:
       counter += 1
       try:
          numeric = [next(training_chars) for _ in range(args.batchSize)]
       except StopIteration:
          break
       printHere = (counter % 500 == 0)
       forward(numeric, printHere=printHere, train=True)
       if printHere:
           print((counter))
           print("Dev losses")
           print(devLosses)
           print("Chars per sec "+str(trainChars/(time.time()-startTime)))
    assert (len(labels)) > 1000
    assert (len(labels)) < 2000
    
    print("Creating regression model")
    
    x_train = hidden_states
    y_train = labels
    
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
     
    
    devLosses = []
    
    for ROUND in range(4):
       test_set = (x for x in test_chars_perRound[ROUND])
      
       
       errors = []
       scores = []
       
       examples_count = 0
       
       correct = 0
       falsePositives = 0
       falseNegatives = 0
       
       devLosses = []
       startTime = time.time()
       trainChars = 0
       counter = 0
       for _ in range(5000):
          hidden_states = []
          labels = []
          relevantWords = []
          relevantNextWords = []
          labels_sum = 0
          boundary_positions = []
          positionIDs = []
          charactersAll = []
          counter += 1
          try:
             numeric = [next(test_set) for _ in range(args.batchSize)]
          except StopIteration:
             break
          printHere = (counter % 500 == 0)
          forward(numeric, printHere=printHere, train=False)
          if printHere:
              print((counter))
              print("Dev losses")
              print(devLosses)
              print("Chars per sec "+str(trainChars/(time.time()-startTime)))
       
          if len(hidden_states) == 0:
               continue
          assert len(hidden_states) > args.batchSize * args.sequence_length * 0.4
          predictors = hidden_states
          dependent = labels
          
          x_test = predictors
          y_test = dependent
          words_test = relevantWords
          #next_words_test = relevantNextWords
          
          
          
          
          predictions = logisticRegr.predict(x_test)
          for ind in range(len(positionIDs)):
             print("\t".join([str(ROUND), str(FOLD)] + [str(x[ind]) for x in [positionIDs, charactersAll if charactersAll != "\n" else "\\n", predictions, y_test]]), file=outFileTable)
          
          score = logisticRegr.score(x_test, y_test)
          scores.append(score)
        
        
          for i in range(len(predictions)):
              if predictions[i] != y_test[i]:
                    if predictions[i] == 1:
                      falsePositives += 1
                    elif predictions[i] == 0:
                      falseNegatives += 1
              else:
                 correct += 1
          print("Balance ",sum(y_test)/len(y_test), ROUND, FOLD)
          examples_count += len(y_test)
          print(correct, falsePositives)   
    
          print("precision", correct / (correct + falsePositives))
          print("recall", correct / (correct + falseNegatives))
    
    
    
