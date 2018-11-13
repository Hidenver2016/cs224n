# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:46:30 2018

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
from collections import Counter, OrderedDict
import nltk
from copy import deepcopy
import os
import re
import unicodedata
flatten = lambda l: [item for sublist in l for item in sublist]

from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
random.seed(1024)
#%matplotlib inline


USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex=0
    eindex=batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
        
#%%padding        
# It is for Sequence 2 Sequence format
def pad_to_batch(batch, x_to_ix, y_to_ix):
    
    sorted_batch =  sorted(batch, key=lambda b:b[0].size(1), reverse=True) # sort by len
    x,y = list(zip(*sorted_batch))
    max_x = max([s.size(1) for s in x])
    max_y = max([s.size(1) for s in y])
    x_p, y_p = [], []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:#                                    '<PAD>' is 0 here
            x_p.append(torch.cat([x[i], Variable(LongTensor([x_to_ix['<PAD>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
        if y[i].size(1) < max_y:
            y_p.append(torch.cat([y[i], Variable(LongTensor([y_to_ix['<PAD>']] * (max_y - y[i].size(1)))).view(1, -1)], 1))
        else:
            y_p.append(y[i])
        
    input_var = torch.cat(x_p) # all x_p is the length of max_x including the added '<Pad>', 0
    target_var = torch.cat(y_p)
    input_len = [list(map(lambda s: s ==0, t.data)).count(False) for t in input_var]# real input data length (without the added '<Pad>')
    target_len = [list(map(lambda s: s ==0, t.data)).count(False) for t in target_var]
    
    return input_var, target_var, input_len, target_len

def prepare_sequence(seq, to_index): # map word into index
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))


#%% Data load and Preprocessing
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

corpus = open('dataset/translate/eng-fra.txt', 'r', encoding='utf-8').readlines()
len(corpus)
corpus = corpus[:30000] # for practice
MIN_LENGTH = 3
MAX_LENGTH = 25

#%%time
X_r, y_r = [], [] # raw

for parallel in corpus:
    so,ta = parallel[:-1].split('\t')# because in windows, the last one is '\n', exclude
    if so.strip() == "" or ta.strip() == "": 
        continue
    
    normalized_so = normalize_string(so).split()
    normalized_ta = normalize_string(ta).split()
    
    if len(normalized_so) >= MIN_LENGTH and len(normalized_so) <= MAX_LENGTH \
    and len(normalized_ta) >= MIN_LENGTH and len(normalized_ta) <= MAX_LENGTH:
        X_r.append(normalized_so)
        y_r.append(normalized_ta)
    

print(len(X_r), len(y_r))
print(X_r[0], y_r[0])

source_vocab = list(set(flatten(X_r))) #4427
target_vocab = list(set(flatten(y_r))) #7705
print(len(source_vocab), len(target_vocab))


source2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
for vo in source_vocab:
    if source2index.get(vo) is None:
        source2index[vo] = len(source2index)
index2source = {v:k for k, v in source2index.items()}

target2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
for vo in target_vocab:
    if target2index.get(vo) is None:
        target2index[vo] = len(target2index)
index2target = {v:k for k, v in target2index.items()}

#%%time
X_p, y_p = [], [] #output of the mapping (word to index) #29828

for so, ta in zip(X_r, y_r):
    X_p.append(prepare_sequence(so + ['</s>'], source2index).view(1, -1)) # add '</s>' as stop sign
    y_p.append(prepare_sequence(ta + ['</s>'], target2index).view(1, -1))
    
train_data = list(zip(X_p, y_p))


#%%Modeling
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size,hidden_size, n_layers=1,bidirec=False):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        if bidirec:
            self.n_direction = 2 
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        else:
            self.n_direction = 1
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
    
    def init_hidden(self, inputs): # self.n_layers 3; self.n_direction 2 (bi-direction RNN); inputs.size(0) 64; self.hidden_size 512
        hidden = Variable(torch.zeros(self.n_layers * self.n_direction, inputs.size(0), self.hidden_size)) # 6, 64, 512
        return hidden.cuda() if USE_CUDA else hidden
    
    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0) # orthogonal initialization, l0 means the 0 layer
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
    
    def forward(self, inputs, input_lengths):# 64*8, 64
        """
        inputs : B, T (LongTensor)
        input_lengths : real lengths of input batch (list)
        """
        hidden = self.init_hidden(inputs) # 6, 64(batch size), 512 (neuron size)
        
        embedded = self.embedding(inputs) # 64, 8, 300 (embedding size)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True) # Two tensor 376*300; 8×1
        outputs, hidden = self.gru(packed, hidden) # output two tensor 376* 1024; 8*1; hidden 6*64*512, a combination of historical state
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # unpack (back to padded)
                
        if self.n_layers > 1:
            if self.n_direction == 2:
                hidden = hidden[-2:] # take the last 2 hidden states
            else:
                hidden = hidden[-1]
        
        return outputs, torch.cat([h for h in hidden], 1).unsqueeze(1)
    

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size #1024
        self.n_layers = n_layers # 1
        
        # Define the layers
        self.embedding = nn.Embedding(input_size, embedding_size) # 7709, 300
        self.dropout = nn.Dropout(dropout_p)
#                       1324 = 300   + 1024                         
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size) # Attention
    
    def init_hidden(self,inputs): # same as encoder.init_hidden, same sequence
        hidden = Variable(torch.zeros(self.n_layers, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden
    
    
    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
        self.attn.weight = nn.init.xavier_uniform(self.attn.weight)
#         self.attn.bias.data.fill_(0)
#                       
    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D;  1, 64, 1024 
        encoder_outputs : B,T,D;           64, 8, 1024
        encoder_maskings : B,T # ByteTensor;   64, 8
        """
        hidden = hidden[0].unsqueeze(2)  # (1,B,D) -> 64,1024 -> (B,D,1) 64, 1024, 1
        
        batch_size = encoder_outputs.size(0) # B 64
        max_len = encoder_outputs.size(1) # T 8                 64 × 8
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1)) # B,T,D -> B*T,D; 512, 1024
        energies = energies.view(batch_size,max_len, -1) # B,T,D;  64, 8, 1024
        attn_energies = energies.bmm(hidden).squeeze(2) # B,T,D * B,D,1 --> B,T;  64, 8
        
#         if isinstance(encoder_maskings,torch.autograd.variable.Variable):
#             attn_energies = attn_energies.masked_fill(encoder_maskings,float('-inf'))#-1e12) # PAD masking
        
        alpha = F.softmax(attn_energies,1) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T  64, 1, 8
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D ; 64, 1, 1024
        
        return context, alpha
    
    
    def forward(self, inputs, context, max_length, encoder_outputs, encoder_maskings=None, is_training=False):
        """
        inputs : B,1 (LongTensor, START SYMBOL) 64, 1
        context : B,1,D (FloatTensor, Last encoder hidden state) 64,1,1024
        max_length : int, max length to decode # for batch
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        is_training : bool, this is because adapt dropout only training step.
        """
        # Get the embedding of the current input word
        embedded = self.embedding(inputs) # 64, 1, 300
        hidden = self.init_hidden(inputs) # 1, 64, 1024
        if is_training:
            embedded = self.dropout(embedded)
        
        decode = []
        # Apply GRU to the output so far
        for i in range(max_length):
#                                            64, 1, 1324                           
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c) 1, 64, 1024
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c), 1, 64, 2048
            score = self.linear(concated.squeeze(0)) # 64*7709
            softmaxed = F.log_softmax(score,1) #64*7709
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1] # 64, for a batch(64), choose the position of highest score for each one
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1} 64*1*300
            if is_training:
                embedded = self.dropout(embedded)
            
            # compute next context vector using attention
            context, alpha = self.Attention(hidden, encoder_outputs, encoder_maskings)
            
        #  column-wise concat, reshape!!
        scores = torch.cat(decode, 1) #64, 77090
        return scores.view(inputs.size(0) * max_length, -1) #640, 7709
    
    def decode(self, context, encoder_outputs):
        start_decode = Variable(LongTensor([[target2index['<s>']] * 1])).transpose(0, 1)
        embedded = self.embedding(start_decode)
        hidden = self.init_hidden(start_decode)
        
        decodes = []
        attentions = []
        decoded = embedded
        while decoded.data.tolist()[0] != target2index['</s>']: # until </s>
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score,1)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            context, alpha = self.Attention(hidden, encoder_outputs,None)
            attentions.append(alpha.squeeze(1))
        
        return torch.cat(decodes).max(1)[1], torch.cat(attentions)


#%%Train
EPOCH = 50
BATCH_SIZE = 64
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 512
LR = 0.001
DECODER_LEARNING_RATIO = 5.0
RESCHEDULED = False
#                      4431               300             512     3 layers
encoder = Encoder(len(source2index), EMBEDDING_SIZE, HIDDEN_SIZE, 3, True)
decoder = Decoder(len(target2index), EMBEDDING_SIZE, HIDDEN_SIZE * 2)
encoder.init_weight()
decoder.init_weight()

if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

loss_function = nn.CrossEntropyLoss(ignore_index=0)
enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)

for epoch in range(EPOCH):
    losses=[]
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        inputs, targets, input_lengths, target_lengths = pad_to_batch(batch, source2index, target2index) # standard inputs (with standard length), real input length (without added '<Pad>')
        # 64*8
        input_masks = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data)))) for t in inputs]).view(inputs.size(0), -1) # label the added <Pad> for inputs
        start_decode = Variable(LongTensor([[target2index['<s>']] * targets.size(0)])).transpose(0, 1) #64*1
        encoder.zero_grad()
        decoder.zero_grad()  #     64,8    64
        output, hidden_c = encoder(inputs, input_lengths) # 64*8*1024; 64*1*1024
        
        preds = decoder(start_decode, hidden_c, targets.size(1), output, input_masks, True) # 640, 7709
                                
        loss = loss_function(preds, targets.view(-1))
#        losses.append(loss.data.tolist()[0] )
        losses.append(loss.data.tolist())
        loss.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), 50.0) # gradient clipping
        torch.nn.utils.clip_grad_norm(decoder.parameters(), 50.0) # gradient clipping
        enc_optimizer.step()
        dec_optimizer.step()

        if i % 200==0:
            print("[%02d/%d] [%03d/%d] mean_loss : %0.2f" %(epoch, EPOCH, i, len(train_data)//BATCH_SIZE, np.mean(losses)))
            losses=[]

    # You can use http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
    if RESCHEDULED == False and epoch  == EPOCH//2:
        LR *= 0.01
        enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)
        RESCHEDULED = True
        

#%%Test
def show_attention(input_words, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     show_plot_visdom()
    plt.show()
    plt.close()
    
test = random.choice(train_data)
input_ = test[0]
truth = test[1]

output, hidden = encoder(input_, [input_.size(1)])
pred, attn = decoder.decode(hidden, output)

input_ = [index2source[i] for i in input_.data.tolist()[0]]
pred = [index2target[i] for i in pred.data.tolist()]


print('Source : ',' '.join([i for i in input_ if i not in ['</s>']]))
print('Truth : ',' '.join([index2target[i] for i in truth.data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ',' '.join([i for i in pred if i not in ['</s>']]))

if USE_CUDA:
    attn = attn.cpu()

show_attention(input_, pred, attn.data)




























