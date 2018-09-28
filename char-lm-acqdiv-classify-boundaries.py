
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--save-to", dest="save_to", type=str)
parser.add_argument("--gpu", dest="gpu", type=bool)


import random

parser.add_argument("--batchSize", type=int, default=64)
parser.add_argument("--char_embedding_size", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=2048)
parser.add_argument("--layer_num", type=int, default=2)
parser.add_argument("--weight_dropout_in", type=float, default=0.3)
parser.add_argument("--weight_dropout_hidden", type=float, default=0.25)
parser.add_argument("--char_dropout_prob", type=float, default=0.05)
parser.add_argument("--char_noise_prob", type = float, default= 0.0)
parser.add_argument("--learning_rate", type = float, default= 0.4)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=50)


args=parser.parse_args()
print(args)


def device(x):
    if args.gpu:
        return x.cuda()
    else:
        return x



from acqdivReader import AcqdivReader, AcqdivReaderPartition

acqdivCorpusReader = AcqdivReader(args.language)



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

try:
   with open("/checkpoint/mhahn/char-vocab-acqdiv-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
    print("Creating new vocab")
    char_counts = {}
    # get symbol vocabulary
    with open("/private/home/mhahn/data/acqdiv/"+args.language+"-vocab.txt", "r") as inFile:
      words = inFile.read().strip().split("\n")
      for word in words:
         for char in word.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
    char_counts = [(x,y) for x, y in char_counts.items()]
    itos = [x for x,y in sorted(char_counts, key=lambda z:(z[0],-z[1]))]
    with open("/checkpoint/mhahn/char-vocab-acqdiv-"+args.language, "w") as outFile:
       print("\n".join(itos), file=outFile)
#itos = sorted(itos)
itos.append("\n")
itos.append(" ")
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])

halfSequenceLength = int(args.sequence_length/2)



import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn = device(torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num))

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)


rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

# -1, because whitespace doesn't actually appear
output = device(torch.nn.Linear(args.hidden_dim, len(itos)-1+3))
char_embeddings = device(torch.nn.Embedding(num_embeddings=len(itos)-1+3, embedding_dim=args.char_embedding_size))

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load("/checkpoint/mhahn/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable

# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
      numeric = [0]
      boundaries = [None for _ in range(args.sequence_length+1)]
      count = 0
      currentWord = ""
      print("Prepare chunks")
      for chunk in data:
       print(len(chunk))
       for char in chunk:
         if char == " ":
           boundaries[len(numeric)] = currentWord
           currentWord = ""
           continue
         count += 1
         currentWord += char
#         if count % 100000 == 0:
#             print(count/len(data))
         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric, boundaries
            numeric = [0]
            boundaries = [None for _ in range(args.sequence_length+1)]


# from each bath element, get one positive example OR one negative example

wordsSoFar = set()
hidden_states = []
labels = []
labels_sum = 0

def forward(numeric, train=True, printHere=False):
      global labels_sum
      numeric, boundaries = zip(*numeric)
#      print(numeric)
 #     print(boundaries)

      input_tensor = Variable(device(torch.LongTensor(numeric).transpose(0,1)[:-1]), requires_grad=False)
      target_tensor = Variable(device(torch.LongTensor(numeric).transpose(0,1)[1:]), requires_grad=False)

      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      out, _ = rnn_drop(embedded, None)
#      if train:
#          out = dropout(out)

      for i in range(len(boundaries)):
         target = (labels_sum + 10 < len(labels)/2) or (random.random() < 0.5)
         #if :
         
#         print(boundaries[i])
#        print(target)
#         print(boundaries[i]) 
         true = sum([((x == None) if target == False else (x not in wordsSoFar)) for x in boundaries[i][int(args.sequence_length/2):-1]])
 #        print(target, true)
         if true == 0:
            continue
         soFar = 0
         for j in range(len(boundaries[i])):
           if j < int(len(boundaries[i])/2):
               continue
           if (lambda x:((x is None if target == False else x not in wordsSoFar)))(boundaries[i][j]):
 #             print(i, target, true,soFar)
              if random.random() < 1/(true-soFar):
                  hidden_states.append(out[j,i].detach().data.cpu().numpy())
                  labels.append(1 if target else 0)
                  labels_sum += labels[-1]
                  if target:
                     wordsSoFar.add(boundaries[i][j])
                  break
              soFar += 1
         assert soFar < true
#      print(hidden_states)
#      print(labels)

      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      loss = train_loss(log_probs.view(-1, len(itos)-1+3), target_tensor.view(-1))

      if printHere:
         lossTensor = print_loss(log_probs.view(-1, len(itos)-1+3), target_tensor.view(-1)).view(args.sequence_length, len(numeric))
         losses = lossTensor.data.cpu().numpy()
#         boundaries_index = [0 for _ in numeric]
         for i in range((args.sequence_length-1)-1):
 #           if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
  #             boundary = True
   #            boundaries_index[0] += 1
    #        else:
     #          boundary = False
            print((losses[i][0], itos[numeric[0][i+1]-3]))
         print((labels_sum, len(labels)))
     # return loss, len(numeric) * args.sequence_length



import time

devLosses = []
#for epoch in range(10000):
if True:

   
   data = AcqdivReaderPartition(acqdivCorpusReader, partition="train").iterator(blankBeforeEOS=False)
#   data = data.reshuffledIterator(blankBeforeEOS=False, originalIterator=AcqdivReader.iteratorMorph)
   

  # training_data = corpusIteratorWiki.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(data, train=True)



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

      if len(labels) > 10000:
         break

predictors = hidden_states
dependent = labels

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, dependent, test_size=0.1, random_state=0, shuffle=True)


from sklearn.linear_model import LogisticRegression

print("regression")

logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)


score = logisticRegr.score(x_test, y_test)
print("Balance ",sum(y_test)/len(y_test))
print(score)






#   dev_data = corpusIteratorWiki.dev(args.language)
#   print("Got data")
#   dev_chars = prepareDataset(dev_data, train=True) if args.language == "italian" else prepareDatasetChunks(dev_data, train=True)
#
#
#     
#   dev_loss = 0
#   dev_char_count = 0
#   counter = 0
#
#   while True:
#       counter += 1
#       try:
#          numeric = [next(dev_chars) for _ in range(args.batchSize)]
#       except StopIteration:
#          break
#       printHere = (counter % 50 == 0)
#       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
#       dev_loss += numberOfCharacters * loss.cpu().data.numpy()[0]
#       dev_char_count += numberOfCharacters
#   devLosses.append(dev_loss/dev_char_count)
#   print(devLosses)
#   with open("/checkpoint/mhahn/"+args.language+"_"+__file__+"_"+str(args.myID), "w") as outFile:
#      print(" ".join([str(x) for x in devLosses]), file=outFile)
#
#   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
#      break
#
