import sys
import numpy as np

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

K = 10

"""read in Hindi numbers 1-99"""
numbers="""e k
d o
t I n
c A r
p An c
ch e
s A t
A Th
n au
d a s
g y A r a
b A r a
t E r a
c au d a
p a n d r a
s o l a
s a t r a
a Th A r a
u n n I s
b I s
i k k I s
b A I s
t E I s
c au b I s
p a c c I s
ch a b b I s
s a t t A I s
a T T A I s
u n t I s
t I s
i k a t t I s
b a t t I s
t ain t I s
c aun t I s
p ain t I s
ch a t t I s
s ain t I s
a R t I s
u n t A l I s
c A l I s
i k t A l I s
b a y A l I s
t ain t A l I s
c a v A l I s
p ain t A l I s
ch i y A l I s
s ain t A l I s
a R t A l I s
u n c A s
p a c A s
i k y A v a n
b A v a n
t i r p a n
c au v a n
p a c p a n
ch a p p a n
s a t t A v a n
a T Th A v a n
u n s a Th
s A Th
i k s a Th
b A s a Th
t i r s a Th
c aun s a Th
p ain s a Th
ch i y A s a Th
s a r s a Th
a R s a Th
u n h a t t a r
s a t t a r
i k h a t t a r
b a h a t t a r
t i h a t t a r
c au h a t t a r
p a c h a t t a r
ch i h a t t a r
s a t h a t t a r
a Th h a t t a r
u n y A s I
a s s I
i k y A s I
b a y A s I
t i r A s I
c au r A s I
p a c A s I
ch i y A s I
s a t t A s I
a T Th A s I
n a v A s I
n a v E
i k y A n v E
b A n v E
t i r A n v E
c au r A n v E
p a c A n v E
ch i y A n v E
s a t t A n v E
a T Th A n v E
n i n y A n v E"""

"""process data"""
numbers = [['<bos>']+l.split()+['<eos>'] for l in numbers.split('\n')]
segs = sorted(set([s for l in numbers for s in l]))

S = len(segs)
T = max([len(l) for l in numbers])
N = len(numbers)

sequences = np.zeros([N,T],dtype=np.int32)
for i,w in enumerate(numbers):
    for j,s in enumerate(w):
        sequences[i,j] = segs.index(s)+1


hidden_dim = 32
embed_dim = 32

enc_in = torch.LongTensor(sequences)
dec_in = torch.LongTensor(sequences[:,:-1])
dec_out = torch.LongTensor(sequences[:,1:])

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder_t = nn.LSTM(embed_dim,hidden_dim,1,bidirectional=True)
        self.encoder_d = nn.LSTM(embed_dim,hidden_dim,1,bidirectional=True)
        self.MLP = nn.Linear(hidden_dim*2,K)
        self.decoder = nn.LSTM(embed_dim+embed_dim+embed_dim,hidden_dim,1)
        self.tens_embed = nn.Embedding(K,embed_dim)
        self.digits_embed = nn.Embedding(K,embed_dim)
        self.token_embed = nn.Embedding(S+1,hidden_dim,padding_idx=0)
        self.emit = nn.Linear(hidden_dim,S+1)
    def encode(self,args):
        enc_in = args
        enc_embedded = self.token_embed(enc_in)
        h_enc_t = self.encoder_t(enc_embedded)[0][:,-1]
        p_z_t = torch.softmax(self.MLP(h_enc_t),-1)
        h_enc_d = self.encoder_d(enc_embedded)[0][:,-1]
        p_z_d = torch.softmax(self.MLP(h_enc_d),-1)
        return(p_z_t,p_z_d)
    def decode(self,args):
        z_t,z_d,dec_in = args
        z_t_embed_rep = self.tens_embed(z_t).unsqueeze(1).repeat(1,T-1,1)
        z_d_embed_rep = self.digits_embed(z_d).unsqueeze(1).repeat(1,T-1,1)
        dec_embedded = self.token_embed(dec_in)
        dec_in_concat = torch.cat([z_t_embed_rep,z_d_embed_rep,dec_embedded],-1)
        h_dec = self.decoder(dec_in_concat)[0]
        p_emit = torch.softmax(nn.Linear(hidden_dim,S+1)(h_dec),-1)
        return(p_emit)


"""version with a uniform prior over class membership"""
@config_enumerate
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.AE = autoencoder()
    def model(self,enc_in,dec_in,dec_out,T,N):
        pyro.module("AE", self.AE)
        theta_t = torch.ones(K)/K
        theta_d = torch.ones(K)/K
        with pyro.iarange('data.loop',N,dim=-1) as i:
            z_t = pyro.sample('z_t',dist.Categorical(theta_t.expand(N,K)))
            z_d = pyro.sample('z_d',dist.Categorical(theta_d.expand(N,K)))
            pi = self.AE.decode([z_t,z_d,dec_in[i]])
            for t in range(T-1):
                pyro.sample('y_{}_{}'.format(i,t),dist.Categorical(pi[:,t,:]),obs=dec_out[i,t])
    def guide(self,enc_in,dec_in,dec_out,T,N):
        with pyro.iarange('data.loop',N,dim=-1) as i:
            p_z_t,p_z_d = self.AE.encode(enc_in[i])
            pyro.sample('z_t',dist.Categorical(p_z_t))
            pyro.sample('z_d',dist.Categorical(p_z_d))


"""version with a dirichlet prior over class membership"""
@config_enumerate
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.AE = autoencoder()
    def model(self,enc_in,dec_in,dec_out,T,N):
        pyro.module("AE", self.AE)
        theta_t = pyro.sample('theta_t',dist.Dirichlet(torch.ones(K)*10))
        theta_d = pyro.sample('theta_d',dist.Dirichlet(torch.ones(K)*10))
        with pyro.iarange('data.loop',N,dim=-1) as i:
            z_t = pyro.sample('z_t',dist.Categorical(theta_t.expand(N,K)))
            z_d = pyro.sample('z_d',dist.Categorical(theta_d.expand(N,K)))
            pi = self.AE.decode([z_t,z_d,dec_in[i]])
            for t in range(T-1):
                pyro.sample('y_{}_{}'.format(i,t),dist.Categorical(pi[:,t,:]),obs=dec_out[i,t])
    def guide(self,enc_in,dec_in,dec_out,T,N):
        with pyro.iarange('data.loop',N,dim=-1) as i:
            p_z_t,p_z_d = self.AE.encode(enc_in[i])
            pyro.sample('z_t',dist.Categorical(p_z_t))
            pyro.sample('z_d',dist.Categorical(p_z_d))


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


gradient_norms = defaultdict(list)
svi.loss(vae.model, vae.model, enc_in,dec_in,dec_out,T,N)  # Initializes param store.
for name, value in pyro.get_param_store().named_parameters():
    value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))
