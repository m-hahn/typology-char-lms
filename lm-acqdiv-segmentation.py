from config import VOCAB_HOME, CHAR_VOCAB_HOME, CHECKPOINT_HOME



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--save-to", dest="save_to", type=str)
parser.add_argument("--gpu", dest="gpu", type=bool)


import random

parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--char_embedding_size", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--layer_num", type=int, default=1)
parser.add_argument("--weight_dropout_in", type=float, default=0.01)
parser.add_argument("--weight_dropout_hidden", type=float, default=0.1)
parser.add_argument("--char_dropout_prob", type=float, default=0.33)
parser.add_argument("--char_noise_prob", type = float, default= 0.01)
parser.add_argument("--learning_rate", type = float, default= 0.1)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=20)


args=parser.parse_args()
print(args)

# For putting things on the GPU if the --gpu flag is set
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

# read the character vocabulary
try:
   with open(CHAR_VOCAB_HOME+"/char-vocab-acqdiv-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError: # or, if that fails, construct one for the language
    print("Creating new vocab")
    char_counts = {}
    # get symbol vocabulary
    with open(VOCAB_HOME+args.language+"-vocab.txt", "r") as inFile:
      words = inFile.read().strip().split("\n")
      for word in words:
         for char in word.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
    char_counts = [(x,y) for x, y in char_counts.items()]
    itos = [x for x,y in sorted(char_counts, key=lambda z:(z[0],-z[1]))]
    with open(CHAR_VOCAB_HOME+"/char-vocab-acqdiv-"+args.language, "w") as outFile:
       print("\n".join(itos), file=outFile)
#itos = sorted(itos)
itos.append("\n")
itos.append(" ")
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
output = torch.nn.Linear(args.hidden_dim, len(itos)-1+3).cuda() 
char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)-1+3, embedding_dim=args.char_embedding_size).cuda()

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

# Read the training data, so that it is reshuffled. When evaluating, we need to switch to a held-out set, but for development of this code, I'm using the training set (i.e., training set of the language model).
data = AcqdivReaderPartition(acqdivCorpusReader, partition="train").reshuffledIterator(blankBeforeEOS=False)


# Read the data into a list of integers
numeric_with_blanks = []
count = 0
print("Prepare chunks")
for chunk in data: # each "chunk" is a string representing an utterance
  numeric_with_blanks.append(stoi[" "]+3) # record a whitespace before each utterance, as an utterance boundary is a word boundary by definition
  for char in chunk:
    count += 1
    if char not in stoi: # print characters that are not in the vocabulary so one can see if there is a systematic coverage issue with the vocabulary
        print("Unseen character", char)
    numeric_with_blanks.append(stoi[char]+3 if char in stoi else 2) # record the character if it is in the vocabulary, otherwise OOV (whose numerical value is 2)

# For speed, only running on a portion of the training set
numeric_with_blanks = numeric_with_blanks[:100000]

# record word boundaries
boundaries = [] # the positions of word boundaries in numeric_full -- that is, the indices of every first character of a word
numeric_full = [] # the corpus without word boundaries ( as a list of integers)
for entry in numeric_with_blanks:
  if entry > 3 and itos[entry-3] == " ": # "entry > 3" means that the entry represents a real character, not OOV or EOS. If this holds, check whether the character is whitespace.
     boundaries.append(len(numeric_full)) # record that the last character recorded in numeric_full so far is at a word boundary
  else:
     numeric_full.append(entry)

# for each position, recording the statistics that word segmemtation is based on
future_surprisal_with = [None for _ in numeric_full]
future_surprisal_without = [None for _ in numeric_full]

char_surprisal = [None for _ in numeric_full]
char_entropy = [None for _ in numeric_full]




###########################################################################################################3
# Run the CNLM and record surprisal and entropy measures.
###########################################################################################################3


# Slice the dataset into minibatches of length args.sequence_length each
for start in range(0, len(numeric_full)-args.sequence_length, args.batchSize):
      numeric = [([0] + numeric_full[b:b+args.sequence_length]) for b in range(start, start+args.batchSize)] # extract a minibatch
      maxLength = max([len(x) for x in numeric])
      for i in range(len(numeric)): # If necessary, pad entries with EOS so that they all have the same length
        numeric[i] = numeric[i] + [0]*(maxLength-len(numeric[i]))

      # Inputs to the LSTM
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)
      embedded = char_embeddings(input_tensor)

      # Run the LSTM on the minibatch
      out, _ = rnn_drop(embedded, None)
      logits = output(out) 
      log_probs = logsoftmax(logits)

      # For each position, compute the entropy over the next character
      entropy = (- log_probs * torch.exp(log_probs)).sum(2).view((maxLength-1), args.batchSize).data.cpu().numpy()

      # For each position, compute the surprisal of the next character.
      loss = print_loss(log_probs.view(-1, len(itos)-1+3), target_tensor.view(-1)).view((maxLength-1), args.batchSize)
      losses = loss.data.cpu().numpy()

      # For 5 % of all minibatches, print the characters and their surprisals to allow sanity-checking
      if random.random() > 0.95:
        print(start/len(numeric_full))
        print(loss.mean())
        for i in range((args.sequence_length-1)-1):
           print((losses[i][0], itos[numeric[0][i+1]-3]))

      # For each element of the minibatch, extract features from the language model
      halfSequenceLength = int(args.sequence_length/2)
      for i in range(start, start+args.batchSize):
         surprisalAtStart = losses[:halfSequenceLength,i-start].sum() # cumulative surprisal of the first half of the sequence
         surprisalAtMid = losses[halfSequenceLength:, i-start].sum() # cumulative surprisal of the second half of the sequence
         if i+halfSequenceLength < len(future_surprisal_with):
            future_surprisal_with[i+halfSequenceLength] = surprisalAtMid # record cumulative surprisal of the second half of the sequence (used for PMI)
            char_surprisal[i+halfSequenceLength] = losses[halfSequenceLength, i-start] # extract surprisal feature for the position at the center of the sequence
            char_entropy[i+halfSequenceLength] = entropy[halfSequenceLength, i-start] # extract entropy feature for the position at the center of the sequence
         if i < len(future_surprisal_without):
            future_surprisal_without[i] = surprisalAtStart # record the cumulative surprisal of the first half of the sequence (used for PMI)
             
def mi(x,y):
  return   x-y if x is not None and y is not None else None


###########################################################################################################3
# Extract the features used for the logistic classifier
###########################################################################################################3

# records the characters in the data, useful for extracting a lexicon
chars = []

# predictor and dependent variable for the logistic regression
predictor = []
dependent = []

utteranceBoundaries = []
lastWasUtteranceBoundary = False

boundaries_index = 0
for i in range(len(numeric_full)): # go through the dataset
   if boundaries_index < len(boundaries) and i == boundaries[boundaries_index]:
      boundary = True # the i-th character is a first character of a word
      boundaries_index += 1
   else:
      boundary = False # the i-th character is not a first character of a word
   pmiFuturePast = mi(future_surprisal_without[i], future_surprisal_with[i]) # compute the PMI feature for the i-th position
   print((itos[numeric_full[i]-3], char_surprisal[i], pmiFuturePast, pmiFuturePast < 0 if pmiFuturePast is not None else None, boundary)) # print some of the features
   if pmiFuturePast is not None:
     character = itos[numeric_full[i]-3] if numeric_full[i] != 2 else itos[-3]
     assert character != " "
     if character == "\n": # This is the end-of-utterance character; it is skipped for the word segmentation classifier but is used to extract a feature indicating utterance boundaries
        lastWasUtteranceBoundary = True # record whether the last word boundary was also an utterance boundary.
     else: # For ordinary characters (phonetic characters, neither end-of-utterance nor whitespace)
       chars.append(character)
       # record the features: PMI, Surprisal, Entropy, and whether the current position is an utterance boundary (since the Bayesian model knows utterance boundaries, giving them as feature here is fair).
       predictor.append([pmiFuturePast, char_surprisal[i], char_entropy[i], 1 if lastWasUtteranceBoundary else 0]) 
       dependent.append(1 if boundary else 0) # the dependent variable: word boundary or not
       lastWasUtteranceBoundary = False


# extract features for the three positions before and after each position
zeroPredictor = [0]*len(predictor[0])

predictorShiftedP1 = predictor[1:]+[zeroPredictor]
predictorShiftedP2 = predictor[2:]+[zeroPredictor,zeroPredictor]
predictorShiftedP3 = predictor[3:]+[zeroPredictor,zeroPredictor,zeroPredictor]

predictorShiftedM1 = [zeroPredictor]+predictor[:-1]
predictorShiftedM2 = [zeroPredictor,zeroPredictor]+predictor[:-2]
predictorShiftedM3 = [zeroPredictor,zeroPredictor,zeroPredictor]+predictor[:-3]

predictor = [a+b+c+d+e+f+g for a, b, c, d, e, f, g in zip(predictor, predictorShiftedP1, predictorShiftedP2, predictorShiftedP3, predictorShiftedM1, predictorShiftedM2, predictorShiftedM3)]



###########################################################################################################3
# Run the logistic classifier
###########################################################################################################3



# Split the dataset into train/test for the logistic classifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, chars_train, chars_test = train_test_split(predictor, dependent, chars, test_size=0.5, random_state=0, shuffle=False)


from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()

# Train the logistic classifier
logisticRegr.fit(x_train, y_train)

# Predict boundaries on the classifier's test set: For each position, predict whether it is the first character of a word
predictions = logisticRegr.predict(x_test)

# Print the test data and predictions for sanity-checking
for char, predicted, real, predictor in zip(chars_test, predictions, y_test, x_test):
    print((char, predicted, real, predictor[0]))


###########################################################################################################3
# Evaluate the resulting segmentations
###########################################################################################################3


realLexicon = set() # the lexicon of real words in the test data
extractedLexicon = {} # extracted words, with their counts
currentWord = ""  # while going through the data, maintain the current real and predicted word
currentWordReal = ""
realWords = 0 # how many tokens there are
predictedWords = 0 # how many tokens are extracted by the model
agreement = 0 # how many tokens are extracted correctly by the model

# go through the test data to evaluate the segmentation
for char, predicted, real in zip(chars_test, predictions, y_test):
   assert char != " "
   # "real" is a boolean indicating whether the current position is a real (ground-truth) word boundary
   # "predicted" is a boolean indicating whether the current position is predicted to a boundary by the classifier
   if real ==1: # there is a ground-truth boundary here
       realWords += 1
       if predicted == 1 and currentWord == currentWordReal: # if the token ending here is correctly extracted also by the model
           agreement += 1
       realLexicon.add(currentWordReal) # store the word
       currentWordReal = char # start recording the next word
   else:
       currentWordReal += char # no boundary here, so continue assembling the current word

   if predicted == 1: # similar to the "real == 1" case: there is a predicted boundary here
       predictedWords += 1
       extractedLexicon[currentWord] = extractedLexicon.get(currentWord, 0) + 1
       currentWord = char # start recording the next predicted word
   else:
       currentWord += char # continue recording the current word

print("Extracted words")
print(sorted(list(extractedLexicon.items()), key=lambda x:x[1]))
print("Incorrect Words")
incorrectWords = [(x,y) for (x,y) in extractedLexicon.items() if x in set(list(extractedLexicon)).difference(realLexicon)]
print(sorted(incorrectWords, key=lambda x:x[1]))
print("Correct words")
correctWords = [(x,y) for (x,y) in extractedLexicon.items() if x in set(list(extractedLexicon)).intersection(realLexicon)]
print(sorted(correctWords, key=lambda x:x[1]))
print("=====Lexicon=====")
print("Precision")
print(len(correctWords)/len(extractedLexicon))
print("Recall")
print(len(correctWords)/len(realLexicon))
print("..")

print("=====Tokens"=====)
print("Precision")
print(agreement/predictedWords)
print("Recall")
print(agreement/realWords)


# Finally, record results on classifying positions for being a boundary or not
predictedBoundariesTotal = 0
predictedBoundariesCorrect = 0
realBoundariesTotal = 0

predictedAndReal = len([1 for x, y in zip(predictions, y_test) if x==1 and x==y])
predictedCount = sum(predictions)
targetCount = sum(y_test)
print("=====Boundaries=====")
print("Precision")
print(predictedAndReal/predictedCount)
print("Recall")
print(predictedAndReal/targetCount)

score = logisticRegr.score(x_test, y_test)
print(score)




