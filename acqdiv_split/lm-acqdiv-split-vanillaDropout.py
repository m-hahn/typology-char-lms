from config import VOCAB_HOME, CHAR_VOCAB_HOME, CHECKPOINT_HOME

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)
parser.add_argument("--out-loss-filename", dest="out_loss_filename", type=str)


import random

parser.add_argument("--batchSize", type=int, default=random.choice([16, 32]))
parser.add_argument("--char_embedding_size", type=int, default=random.choice([50, 100]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([512, 1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([1,2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05,0.3,0.5]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05, 0.3, 0.5]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.05]))
parser.add_argument("--char_noise_prob", type = float, default=0)
parser.add_argument("--learning_rate", type = float, default=random.choice([0.1, 1, 2, 5, 10]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=40)


args=parser.parse_args()
print(args)


from acqdivReadersplit import AcqdivReader, AcqdivReaderPartition

acqdivCorpusReadertrain = AcqdivReader("train", args.language)
acqdivCorpusReaderdev = AcqdivReader("dev", args.language)
acqdivCorpusReadertest = AcqdivReader("test", args.language)


def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

itos=[]
with open(VOCAB_HOME + args.language + '-char.txt', "r") as inFile:
     for line in inFile:
      line=line.strip()
      itos.append(line)
itos.append("\n")
stoi = dict([(itos[i],i) for i in range(len(itos))])

print(itos)
print(stoi)


import random


import torch

print(torch.__version__)



rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)


#rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

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
   checkpoint = torch.load(CHECKPOINT_HOME+args.load_from+".pth.tar")
   for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)])

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
   numeric = [0]
   count = 0
   print("Prepare chunks")
   for chunk in data:
      #       print(len(chunk))
      for char in chunk:
         if char == ";":
            continue
         count += 1
         #         if count % 100000 == 0:
         #             print(count/len(data))
         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric
            numeric = [0]




def prepareDataset(data, train=True):
   numeric = [0]
   count = 0
   for char in data:
      if char == ";":
         continue
      count += 1
      #         if count % 100000 == 0:
      #             print(count/len(data))
      numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
      if len(numeric) > args.sequence_length:
         yield numeric
         numeric = [0]

# Dropout Masks
bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * (args.sequence_length) * args.char_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * (args.sequence_length) * args.hidden_dim)]).cuda())



def forward(numeric, train=True, printHere=False):
    input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
    target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)

    #  print(char_embeddings)
    #if train and (embedding_full_dropout_prob is not None):
    #   embedded = embedded_dropout(char_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #char_embeddings(input_tensor)
    #else:
    embedded = char_embeddings(input_tensor)
    if train:
       embedded = char_dropout(embedded)
       mask = bernoulli_input.sample()
       mask = mask.view(args.sequence_length, args.batchSize, args.char_embedding_size)
       embedded = embedded * mask

    out, _ = rnn(embedded, None)
    #      if train:
    #          out = dropout(out)

    if train:
      mask = bernoulli_output.sample()
      mask = mask.view(args.sequence_length, args.batchSize, args.hidden_dim)
      out = out * mask





    logits = output(out)
    log_probs = logsoftmax(logits)
    #   print(logits)
    #    print(log_probs)
    #     print(target_tensor)

    loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

    if printHere:
       lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(args.sequence_length, len(numeric))
       losses = lossTensor.data.cpu().numpy()
       #         boundaries_index = [0 for _ in numeric]
       for i in range((args.sequence_length-1)-1):
          #           if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
          #             boundary = True
          #            boundaries_index[0] += 1
          #        else:
          #          boundary = False
          target = numeric[0][i+1]-3
          print((losses[i][0], itos[target] if target >= 0 else "OOV"))
    return loss, len(numeric) * args.sequence_length

def backward(loss, printHere):
   optim.zero_grad()
   if printHere:
      print(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_cached, 5.0) #, norm_type="inf")
      optim.step()

import time

devLosses = []
for epoch in range(100):
   print(epoch)
   training_data = AcqdivReaderPartition(acqdivCorpusReadertrain).reshuffledIterator()
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)

   rnn.train(True)
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
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
      trainChars += charCounts
      if printHere:
         print((epoch,counter))
         print("Dev losses")
         print(devLosses)
         print("Chars per sec "+str(trainChars/(time.time()-startTime)))
      if counter % 20000 == 0 and epoch == 0:
        if args.save_to is not None:
           torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), CHECKPOINT_HOME+args.save_to+".pth.tar")


   rnn.train(False)


   dev_data = AcqdivReaderPartition(acqdivCorpusReaderdev).iterator()
   print("Got data")
   dev_chars = prepareDatasetChunks(dev_data, train=True)



   dev_loss = 0
   dev_char_count = 0
   counter = 0

   while True:
      counter += 1
      try:
          numeric = [next(dev_chars) for _ in range(args.batchSize)]
      except StopIteration:
          break
      printHere = (counter % 50 == 0)
      loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
      dev_loss += numberOfCharacters * loss.cpu().data.numpy()
      dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   with open(CHECKPOINT_HOME+args.language+"_"+os.path.basename(__file__)+"_"+str(args.myID), "w") as outFile:
      outFile.write(" ".join([str(x) for x in devLosses]) + '\n')
   if  len(devLosses)>1 and devLosses[-2] < devLosses [-1]:
      min_loss="minimum loss=" + str(float(devLosses[-2])) +" epoch=" + str(epoch-1) + " args=" + str(args)
      break
   else:
      min_loss="minimum loss=" + str(float(devLosses[-1])) +" epoch=" + str(epoch) + " args=" + str(args)
 
   if len(devLosses) > 1 and (devLosses[-1] > devLosses[-2] or devLosses[-1] > 20):
      break
   if args.save_to is not None:
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), CHECKPOINT_HOME+args.save_to+".pth.tar")

print(devLosses)
print(min_loss)

if args.out_loss_filename is not None:
   with open(args.out_loss_filename, 'w') as out_loss_file:
          out_loss_file.write(min_loss + '\n')



	

