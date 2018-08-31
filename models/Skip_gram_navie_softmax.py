# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:12:40 2018

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


#%%
#Data load and Preprocessing

nltk.corpus.gutenberg.fileids()

corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:100] # sampling sentences for test
corpus = [[word.lower() for word in sent] for sent in corpus]


word_count = Counter(flatten(corpus))
border = int(len(word_count) * 0.01) 

stopwords = word_count.most_common()[:border] + list(reversed(word_count.most_common()))[:border]

stopwords = [s[0] for s in stopwords]

stopwords

vocab = list(set(flatten(corpus)) - set(stopwords))
vocab.append('<UNK>')

print(len(set(flatten(corpus))), len(vocab))

word2index = {'<UNK>' : 0} 

for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)

index2word = {v:k for k, v in word2index.items()} 

WINDOW_SIZE = 3
windows = flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])

windows[0]

train_data = []

for window in windows:
    for i in range(WINDOW_SIZE * 2 + 1):
        if i == WINDOW_SIZE or window[i] == '<DUMMY>': 
            continue
        train_data.append((window[WINDOW_SIZE], window[i]))

print(train_data[:WINDOW_SIZE * 2])

X_p = []
y_p = []

train_data[0]

for tr in train_data:
    X_p.append(prepare_word(tr[0], word2index).view(1, -1))
    y_p.append(prepare_word(tr[1], word2index).view(1, -1))
    
train_data = list(zip(X_p, y_p))


len(train_data)

#%%
#Modeling
class Skipgram(nn.Module):
    
    def __init__(self, vocab_size, projection_dim):#len(word2index), EMBEDDING_SIZE(30)
        super(Skipgram,self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim) 
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)

        self.embedding_v.weight.data.uniform_(-1, 1) # init
        self.embedding_u.weight.data.uniform_(0, 0) # init
        #self.out = nn.Linear(projection_dim,vocab_size)
    def forward(self, center_words,target_words, outer_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        outer_embeds = self.embedding_u(outer_words) # B x V x D
        
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1xD * BxDx1 => Bx1
        norm_scores = outer_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # BxVxD * BxDx1 => BxV
        
        nll = -torch.mean(torch.log(torch.exp(scores)/torch.sum(torch.exp(norm_scores), 1).unsqueeze(1))) # log-softmax
        
        return nll # negative log likelihood
    
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        
        return embeds 
    
    
#Train
EMBEDDING_SIZE = 30
BATCH_SIZE = 256
EPOCH = 100

losses = []
model = Skipgram(len(word2index), EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(EPOCH):
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        
        inputs, targets = zip(*batch)
        
        inputs = torch.cat(inputs) # B x 1, 256*1
        targets = torch.cat(targets) # B x 1
        vocabs = prepare_sequence(list(vocab), word2index).expand(inputs.size(0), len(vocab))  # B x V, 256*583
        model.zero_grad()

        loss = model(inputs, targets, vocabs)
        
        loss.backward()
        optimizer.step()
   
#        losses.append(loss.data.tolist()[0])
        losses.append(loss.data.tolist())

    if epoch % 10 == 0:
        print("Epoch : %d, mean_loss : %.02f" % (epoch,np.mean(losses)))
        losses = []
        


#Test
def word_similarity(target, vocab):
    if USE_CUDA:
        target_V = model.prediction(prepare_word(target, word2index))
    else:
        target_V = model.prediction(prepare_word(target, word2index))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target: continue
        
        if USE_CUDA:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        else:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0] 
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10] # sort by similarity

test = random.choice(list(vocab))
print(test)
print(word_similarity(test, vocab))

