import sys
sys.path.append('../models')
from data_process import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

inventory, phone2ix, ix2phone, training_data, dev = process_data(get_corpus_data('../data/hindi/mcgregor.csv'),'cuda')

training_data = np.array(training_data)
encoder_input = torch.LongTensor(training_data)
decoder_input = torch.LongTensor(training_data[:,:-1])
decoder_output = torch.LongTensor(training_data[:,1:])
decoder_output = F.one_hot(decoder_output)[:,:,1:]

N = training_data.shape[0]
T = training_data.shape[1]
S = len(inventory)

K = 10

hidden_dim = 128
embed_dim = 128

class discreteAE(nn.Module):
    def __init__(self):
        super(discreteAE,self).__init__()
        self.embed_enc = nn.Embedding(S+1,embed_dim,padding_idx=0)
        self.encoder = nn.LSTM(embed_dim,hidden_dim,1,bidirectional=True)
        self.V = nn.Linear(hidden_dim*2,K)
        self.embed_dec = nn.Embedding(S+1,embed_dim,padding_idx=0)
        self.decoder = [nn.LSTM(embed_dim,hidden_dim,1) for k in range(K)]
        self.W = nn.Linear(hidden_dim,S-1)
    def forward(self,args):
        encoder_input,decoder_input,decoder_output = args
        enc_embedded = self.embed_enc(encoder_input)
        h_enc = self.encoder(enc_embedded)[0][:,-1,:]
        log_p_z = F.log_softmax(self.V(h_enc),-1)
        dec_embedded = self.embed_dec(decoder_input)
        h_dec = torch.stack([d(dec_embedded)[0] for d in self.decoder],-3)
        log_p_out = torch.sum(F.log_softmax(self.W(h_dec),-1)*decoder_output.unsqueeze(1),[-2,-1])
        losses_z = log_p_z+log_p_out
        losses = torch.logsumexp(losses_z,-1)
        return(-torch.mean(losses),losses_z)

model_loss = discreteAE()

learning_rate = 1e-3
batch_size = 32
epochs = 100

PATH = 'weights_ckpt.pt'
optimizer = torch.optim.Adam(model_loss.parameters(), lr=learning_rate)

idx = np.arange(N)
np.random.shuffle(idx)
for epoch in range(epochs):
    epoch_losses = []
    for (i,j) in list(zip(list(range(0,N,batch_size)),list(range(batch_size,N,batch_size))+[N])):
        batch_idx = idx[i:j]
        X = [encoder_input[batch_idx],decoder_input[batch_idx],decoder_output[batch_idx]]
        loss = model_loss(X)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        epoch_losses.append(loss)
    print("epoch mean loss: {}".format(np.mean(epoch_losses)))
    #torch.save(model_loss.state_dict(), PATH)
    torch.save(model_loss.state_dict(), 'checkpoints/weights_{}.pt'.format(epoch))


#checkpoint = torch.load(PATH)
#model_loss.load_state_dict(checkpoint)