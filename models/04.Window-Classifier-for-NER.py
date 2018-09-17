# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:17:48 2018

@author: hjiang
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
flatten = lambda l: [item for sublist in l for item in sublist]
from sklearn_crfsuite import metrics
random.seed(1024)

print(torch.__version__)
print(nltk.__version__)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))

def prepare_tag(tag,tag2index):
    return Variable(LongTensor([tag2index[tag]]))

#%% Data load and Preprocessing
corpus = nltk.corpus.conll2002.iob_sents()

data = []
for cor in corpus:
    sent, _, tag = list(zip(*cor))
    data.append([sent, tag])
    
print(len(data))
print(data[0])
# build vocab
sents,tags = list(zip(*data))
vocab = list(set(flatten(sents)))
tagset = list(set(flatten(tags)))

word2index={'<UNK>' : 0, '<DUMMY>' : 1} # dummy token is for start or end of sentence
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
index2word = {v:k for k, v in word2index.items()}

tag2index = {}
for tag in tagset:
    if tag2index.get(tag) is None:
        tag2index[tag] = len(tag2index)
index2tag={v:k for k, v in tag2index.items()}

#Prepare Data
WINDOW_SIZE = 2
windows = []

for sample in data:
    dummy = ['<DUMMY>'] * WINDOW_SIZE
    window = list(nltk.ngrams(dummy + list(sample[0]) + dummy, WINDOW_SIZE * 2 + 1))
    windows.extend([[list(window[i]), sample[1][i]] for i in range(len(sample[0]))])
    
windows[0]

len(windows)

random.shuffle(windows)

train_data = windows[:int(len(windows) * 0.9)]
test_data = windows[int(len(windows) * 0.9):]

# Modeling    
class WindowClassifier(nn.Module): 
    def __init__(self, vocab_size, embedding_size, window_size, hidden_size, output_size):

        super(WindowClassifier, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.h_layer1 = nn.Linear(embedding_size * (window_size * 2 + 1), hidden_size)
        self.h_layer2 = nn.Linear(hidden_size, hidden_size)
        self.o_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, inputs, is_training=False): 
        embeds = self.embed(inputs) # BxWxD
        concated = embeds.view(-1, embeds.size(1)*embeds.size(2)) # Bx(W*D)
        h0 = self.relu(self.h_layer1(concated))
        if is_training:
            h0 = self.dropout(h0)
        h1 = self.relu(self.h_layer2(h0))
        if is_training:
            h1 = self.dropout(h1)
        out = self.softmax(self.o_layer(h1))
        return out
    
    
BATCH_SIZE = 128
EMBEDDING_SIZE = 50 # x (WINDOW_SIZE*2+1) = 250
HIDDEN_SIZE = 300
EPOCH = 3
LEARNING_RATE = 0.001

#%%Training
model = WindowClassifier(len(word2index), EMBEDDING_SIZE, WINDOW_SIZE, HIDDEN_SIZE, len(tag2index))
if USE_CUDA:
    model = model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):
    losses = []
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        x,y=list(zip(*batch))
        inputs = torch.cat([prepare_sequence(sent, word2index).view(1, -1) for sent in x])
        targets = torch.cat([prepare_tag(tag, tag2index) for tag in y])
        model.zero_grad()
        preds = model(inputs, is_training=True)
        loss = loss_function(preds, targets)
#        losses.append(loss.data.tolist()[0])
        losses.append(loss.data.tolist())
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print("[%d/%d] mean_loss : %0.2f" %(epoch, EPOCH, np.mean(losses)))
            losses = []
            
            
#%%Test
for_f1_score = []
accuracy = 0
for test in test_data:
    x, y = test[0], test[1]
    input_ = prepare_sequence(x, word2index).view(1, -1)

    i = model(input_).max(1)[1]
    pred = index2tag[i.data.tolist()[0]]
    for_f1_score.append([pred, y])
    if pred == y:
        accuracy += 1

print(accuracy/len(test_data) * 100)

# Print Confusion Matrix
y_pred, y_test = list(zip(*for_f1_score))

sorted_labels = sorted(
    list(set(y_test) - {'O'}),
    key=lambda name: (name[1:], name[0])
)

sorted_labels

y_pred = [[y] for y in y_pred] # this is because sklearn_crfsuite.metrics function flatten inputs
y_test = [[y] for y in y_test]

print(metrics.flat_classification_report(
    y_test, y_pred, labels = sorted_labels, digits=3
))




    












































        