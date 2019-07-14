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



def prepareDatasetChunksTest(data, train=True):
      numeric = [0]
      boundaries = [None for _ in range(args.sequence_length+1)]
      boundariesAll = [None for _ in range(args.sequence_length+1)]

      count = 0
      currentWord = ""
      print("Prepare chunks")
      for chunk in data:
       print(len(chunk))
       for char in chunk:
         if char == ";":
           boundaries[len(numeric)] = currentWord
           boundariesAll[len(numeric)] = currentWord

           currentWord = ""
           continue
         else:
           if boundariesAll[len(numeric)] is None:
               boundariesAll[len(numeric)] = currentWord

         count += 1
         currentWord += char
         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric, boundaries, boundariesAll
            numeric = [0] + numeric[args.sequence_length:]
            assert len(boundaries) == args.sequence_length+1
            boundaries = boundaries[int(args.sequence_length/2):] + [None for _ in range(int(args.sequence_length/2))]
            boundariesAll = boundariesAll[int(args.sequence_length/2):] + [None for _ in range(int(args.sequence_length/2))]
            assert len(boundaries) == args.sequence_length+1




wordsSoFar = set()
hidden_states = []
labels = []
relevantWords = []
relevantNextWords = []
labels_sum = 0

boundary_positions = []

def forward(numeric, train=True, printHere=False):
      global labels_sum
      numeric, boundaries, boundariesAll = zip(*numeric)

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
                 if nextRelevantWord == "END_OF_SEQUENCE":
                    continue
                 target = 1 if boundaries[j][i+1] is not None else 0
                 if train and abs(target+labels_sum - len(labels)/2) > 2:
                    continue
                 hidden_states.append((hidden[1][:,j,:].flatten()[neuron[0]]).unsqueeze(0).cpu().detach().numpy())
                 boundary_positions.append((j,i))
#                 hidden_states.append((hidden[1][:,j,:].flatten()[neuron]).cpu().detach().numpy())

                 labels.append(target)
                 labels_sum += labels[-1]

                 relevantWords.append(boundariesAll[j][i+1])
                 
                 relevantNextWords.append(nextRelevantWord)
                 assert boundariesAll[j][i+1] is not None
                 if j == 0:
                   print(hidden_states[-1], labels[-1], relevantWords[-1], relevantNextWords[-1])  
                 if (labels[-1] == 0) and not relevantNextWords[-1].startswith(relevantWords[-1]):
                     assert False, (relevantWords[-1], relevantNextWords[-1], list(zip(boundaries[j][i:], boundariesAll[j][i:])))
                 if (labels[-1] == 1) and relevantNextWords[-1].startswith(relevantWords[-1]): # this is actually not a hard assertion, it should just be quite unlikely in languages such as English
                     print("WARNING", list(zip(boundaries[j][i:], boundariesAll[j][i:])))
#                     if len(relevantWords[-1]) > 1:
 #                       assert False 

import time

devLosses = []
#for epoch in range(10000):
if True:
   training_data = AcqdivReaderPartition(acqdivCorpusReadertrain).reshuffledIterator(blankBeforeEOS=False)

   print("Got data")
   training_chars = prepareDatasetChunksTest(training_data, train=True)



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
      printHere = (counter % 50 == 0)
      forward(numeric, printHere=printHere, train=True)
      #backward(loss, printHere)
      if printHere:
          print((counter))
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))

      if len(labels) > 1000:
         break
  

predictors = hidden_states #torch.Tensor(hidden_states)
dependent = labels #torch.Tensor(labels).unsqueeze(1)


TEST_FRACTION = 0.0

print("Creating regression model")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, words_train, words_test, next_words_train, next_words_test = train_test_split(predictors, dependent, relevantWords, relevantNextWords, test_size=TEST_FRACTION, random_state=random.randint(1,100), shuffle=True)


from sklearn.linear_model import LogisticRegression

print("regression")

logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)


errors = []
scores = []

examples_count = 0

correct = 0
falsePositives = 0
falseNegatives = 0

with open("segmentation-predictions/"+args.language+"-real.txt", "w") as outFileReal:
  with open("segmentation-predictions/"+args.language+"-predicted.txt", "w") as outFilePredicted:
    
     devLosses = []
     #for epoch in range(10000):
     if True:
     

        rnn_drop.train(False)
        startTime = time.time()
        trainChars = 0
        counter = 0
        for _ in range(10):
           hidden_states = []
           labels = []
           relevantWords = []
           relevantNextWords = []
           labels_sum = 0
           boundary_positions = []

           counter += 1
           try:
              numeric = [next(training_chars) for _ in range(args.batchSize)]
           except StopIteration:
              break
           printHere = (counter % 50 == 0)
           forward(numeric, printHere=printHere, train=False)
           #backward(loss, printHere)
           if printHere:
               print((counter))
               print("Dev losses")
               print(devLosses)
               print("Chars per sec "+str(trainChars/(time.time()-startTime)))
     
           #if len(labels) > 10000:
          #    break
           if len(hidden_states) == 0:
                continue
           predictors = hidden_states
           dependent = labels
           
           x_test = predictors
           y_test = dependent
           words_test = relevantWords
           next_words_test = relevantNextWords
           
           
           
           
           predictions = logisticRegr.predict(x_test)
           
           
           score = logisticRegr.score(x_test, y_test)
           scores.append(score)
      
           parts = [[] for x in range(args.batchSize)]
           
      
           for i in range(len(x_test)):
               parts[boundary_positions[i][0]].append((boundary_positions[i][1], y_test[i], predictions[i], words_test[i], next_words_test[i]))
           for p in parts:
              p = sorted(p, key=lambda x:x[0])
           #   print(p)
              pred_string = "".join([c[3][-1]+(" " if c[2] == 1 else "") for c in p])
              real_string = "".join([c[3][-1]+(" " if c[1] == 1 else "") for c in p])

              print("=========================")
              print(real_string, file=outFileReal)
              print(pred_string, file=outFilePredicted)
      #         print(i, y_test[i], predictions[i], words_test[i], next_words_test[i], boundary_positions[i])
      
      
           for i in range(len(predictions)):
               if predictions[i] != y_test[i]:
                     errors.append((y_test[i], (words_test[i], next_words_test[i], predictions[i], y_test[i])))
                     if predictions[i] == 1:
                       falsePositives += 1
                     elif predictions[i] == 0:
                       falseNegatives += 1
               else:
                  correct += 1
           print("Balance ",sum(y_test)/len(y_test))
           examples_count += len(y_test)
 #       if len(hidden_states) == 0:
#                break


#correct = 0
#falsePositives = 0
#falseNegatives = 0
precision = correct / (correct + falsePositives)
recall = correct / (correct + falseNegatives)

print("Boundary measures", "Precision", precision, "Recall", recall, "F1", 2*(precision*recall)/(precision+recall))


print(examples_count)
score = sum(scores)/len(scores)
print(score)

