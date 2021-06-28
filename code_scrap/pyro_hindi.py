import sys
sys.path.append('../models')
from data_process import *

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO,TraceEnum_ELBO
from pyro.contrib.autoguide import AutoDelta,AutoGuide,AutoContinuous,AutoDiagonalNormal
from pyro.infer import config_enumerate
from torch.distributions import constraints
import time
from collections import defaultdict

inventory, phone2ix, ix2phone, training_data, dev = process_data(get_corpus_data('../data/hindi/mcgregor.csv'),'cuda')

K = 10

S = len(inventory)
T = training_data.shape[1]
N = training_data.shape[0]

hidden_dim = 32
embed_dim = 32
minibatch_size = 32

enc_in = torch.LongTensor(training_data)
dec_in = torch.LongTensor(training_data[:,:-1])
dec_out = torch.LongTensor(training_data[:,1:])

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder = nn.LSTM(embed_dim,hidden_dim,1,bidirectional=True)
        self.MLP = nn.Linear(hidden_dim*2,K)
        self.decoder = nn.LSTM(embed_dim+embed_dim,hidden_dim,1)
        self.embed = nn.Embedding(K,embed_dim)
        self.token_embed = nn.Embedding(S+1,hidden_dim,padding_idx=0)
        self.emit = nn.Linear(hidden_dim,S+1)
    def encode(self,args):
        enc_in = args
        enc_embedded = self.token_embed(enc_in)
        h_enc = self.encoder(enc_embedded)[0][:,-1]
        p_z = torch.softmax(self.MLP(h_enc),-1)
        return(p_z)
    def decode(self,args):
        z,dec_in = args
        z_embed_rep = self.embed(z).unsqueeze(1).repeat(1,T-1,1)
        dec_embedded = self.token_embed(dec_in)
        dec_in_concat = torch.cat([z_embed_rep,dec_embedded],-1)
        h_dec = self.decoder(dec_in_concat)[0]
        p_emit = torch.softmax(nn.Linear(hidden_dim,S+1)(h_dec),-1)
        return(p_emit)

"""version with a dirichlet prior over class membership"""
@config_enumerate
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.AE = autoencoder()
    def model(self,enc_in,dec_in,dec_out,T,N):
        pyro.module("AE", self.AE)
        theta = pyro.sample('theta',dist.Dirichlet(torch.ones(K)*10))
        with pyro.iarange('data.loop',N,dim=-1) as i:
        #with pyro.iarange('data.loop',N,dim=-1,subsample_size=minibatch_size) as i:
            z = pyro.sample('z',dist.Categorical(theta.expand(N,K)))
            pi = self.AE.decode([z,dec_in[i]])
            for t in range(T-1):
                pyro.sample('y_{}_{}'.format(i,t),dist.Categorical(pi[:,t,:]),obs=dec_out[i,t])
    def guide(self,enc_in,dec_in,dec_out,T,N):
        with pyro.iarange('data.loop',N,dim=-1) as i:
            p_z = self.AE.encode(enc_in[i])
            pyro.sample('z',dist.Categorical(p_z))



"""to do: convert to minibatch. It may be necessary to create minibatches externally and pass them to svi.step() rather than putting minibatch statements into VAE() using pyro.iarange(subsample_size)"""
vae = VAE()
optimizer = Adam({"lr": .1})
svi = SVI(vae.model, vae.guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1,strict_enumeration_warning=True))
losses = defaultdict(list)
n_steps = 10000
for c in range(3):
    pyro.clear_param_store()
    for step in range(n_steps):
        start_time = time.time()
        print(step,end=' ')
        loss = svi.step(enc_in,dec_in,dec_out,T,N)
        print(loss,time.time() - start_time)
        losses[c].append(loss)