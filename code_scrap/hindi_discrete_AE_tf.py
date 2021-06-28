import sys
sys.path.append('../models')
from data_process import *

import numpy as np
from collections import defaultdict
import tensorflow              #version '1.12.0'
tf = tensorflow.compat.v1      #if tf.__version__ >= 2.0
from tensorflow.compat.v1.keras.layers import Dense,LSTM,Bidirectional,dot
import tensorflow_probability as tfp #version '0.5.0'
tfd = tfp.distributions
tfb = tfp.bijectors
import time
tf.disable_eager_execution()

inventory, phone2ix, ix2phone, training_data, dev = process_data(get_corpus_data('../data/hindi/mcgregor.csv'),'cuda')

training_data = np.array(training_data)-1
enc_in = training_data
dec_in = training_data[:,:-1]
dec_out = training_data[:,1:]

N = training_data.shape[0]
T = training_data.shape[1]
S = len(inventory)

def LSTM(inputs,weights):
    """standard LSTM"""
    w_kernel = weights['w_kernel']
    w_recurrent = weights['w_recurrent']
    w_bias = weights['w_bias']
    T = inputs.shape[-2]
    H = []
    for t in range(T):
        if t > 0:
            z = tf.einsum('nx,xj->nj',inputs[:,t,:],w_kernel) + tf.einsum('nl,lj->nj',h,w_recurrent) + tf.expand_dims(w_bias,0)
        else:
            z = tf.einsum('nx,xj->nj',inputs[:,t,:],w_kernel) + tf.expand_dims(w_bias,0)
        i,f,o,u = tf.split(z,4,axis=-1)
        i = tf.sigmoid(i)        #input gate
        f = tf.sigmoid(f + 1.0)  #forget gate
        o = tf.sigmoid(o)        #output gate
        u = tf.tanh(u)           #information let in by input gate
        if t > 0:
            c = f * c + i * u
        else:
            c = i * u
        h = o * tf.tanh(c)
        H.append(h)
    H = tf.stack(H,-2)
    return(H)

def MultiLSTM(inputs,weights):
    """run multiple LSTMs in parallel"""
    w_kernel = weights['w_kernel']
    w_recurrent = weights['w_recurrent']
    w_bias = weights['w_bias']
    T = inputs.shape[-2]
    H = []
    for t in range(T):
        if t > 0:
            z = tf.einsum('knx,xj->knj',inputs[:,:,t,:],w_kernel) + tf.einsum('knl,lj->knj',h,w_recurrent) + tf.expand_dims(w_bias,0)
        else:
            z = tf.einsum('knx,xj->knj',inputs[:,:,t,:],w_kernel) + tf.expand_dims(w_bias,0)
        i,f,o,u = tf.split(z,4,axis=-1)
        i = tf.sigmoid(i)        #input gate
        f = tf.sigmoid(f + 1.0)  #forget gate
        o = tf.sigmoid(o)        #output gate
        u = tf.tanh(u)           #information let in by input gate
        if t > 0:
            c = f * c + i * u
        else:
            c = i * u
        h = o * tf.tanh(c)
        H.append(h)
    H = tf.stack(H,-2)
    return(H)

def BiLSTM(inputs,weights_fwd,weights_bkwd):
    """birectional LSTM"""
    forward = LSTM(inputs,weights_fwd)
    backward = tf.reverse(LSTM(tf.reverse(inputs,[-2]),weights_bkwd),[-2])
    return(tf.concat([forward,backward],-1))

K = 8  #number of components
D = 32  #hidden layer dim
J = 32 #embedding dim

batch_idx = tf.placeholder(tf.int32,shape=(None,),name='batch_idx')

def gen_encoder_params():
    """generate parameters for BiLSTM"""
    dims = {
        'enc_fwd':(S,D),
        'enc_bkwd':(S,D)
    }
    LSTM_encoder_params = {}
    for k in dims.keys():
        x = dims[k][0]
        d = dims[k][1]
        LSTM_encoder_params[k] = {}
        LSTM_encoder_params[k]['w_kernel'] = tf.get_variable(name='{}_kernel'.format(k),shape=(x,d*4))
        LSTM_encoder_params[k]['w_recurrent'] = tf.get_variable(name='{}_recurrent'.format(k),shape=(d,d*4))
        LSTM_encoder_params[k]['w_bias'] = tf.get_variable(name='{}_bias'.format(k),shape=(d*4))
    return(LSTM_encoder_params)

def gen_decoder_params():
    """generate parameters for multiple decoder LSTMs, 1 for each component"""
    dims = {
        'dec_fwd':(S+J,D)
    }
    LSTM_decoder_params = {}
    for k in dims.keys():
        x = dims[k][0]
        d = dims[k][1]
        LSTM_decoder_params[k] = {}
        LSTM_decoder_params[k]['w_kernel'] = tf.get_variable(name='{}_kernel'.format(k),shape=(x,d*4))
        LSTM_decoder_params[k]['w_recurrent'] = tf.get_variable(name='{}_recurrent'.format(k),shape=(d,d*4))
        LSTM_decoder_params[k]['w_bias'] = tf.get_variable(name='{}_bias'.format(k),shape=(d*4))
    return(LSTM_decoder_params)
    

def gen_weights():
    dims = {
        'W':(D*2,K),
        'U':(K,J),
        'V':(D,S)
        }
    params = {}
    for k in dims.keys():
        params[k] = tf.get_variable(name=k,shape=dims[k])
    return(params)


LSTM_encoder_params = gen_encoder_params()
LSTM_decoder_params = gen_decoder_params()
params = gen_weights()

def loglik():
    """load batched data, convert to one hot"""
    enc_in_ = tf.one_hot(tf.gather(enc_in,batch_idx),S)
    dec_in_ = tf.one_hot(tf.gather(dec_in,batch_idx),S)
    dec_out_ = tf.one_hot(tf.gather(dec_out,batch_idx),S)
    """mask for decoder input"""
    dec_mask = tf.expand_dims(tf.expand_dims(tf.reduce_sum(dec_in_,-1),-1),1)
    """read input with encoder"""
    h_enc = BiLSTM(enc_in_,LSTM_encoder_params['enc_fwd'],LSTM_encoder_params['enc_bkwd'])
    h_enc_last_state = h_enc[:,-1,:]
    """generate log probability of component membership from last LSTM state"""
    log_p_z = tf.nn.log_softmax(tf.einsum('nd,dk->nk',h_enc_last_state,params['W']),-1)
    """concatenate K component-level embeddings to decoder input"""
    group_embedding = tf.tile(tf.expand_dims(tf.expand_dims(params['U'],1),0),[tf.shape(batch_idx)[0],1,T-1,1])
    dec_in_embedded = tf.concat([group_embedding*dec_mask,tf.tile(tf.expand_dims(dec_in_,1),[1,K,1,1])],-1)
    """read decoder input into K-dimensional decoder"""
    h_dec = MultiLSTM(dec_in_embedded,LSTM_decoder_params['dec_fwd'])
    log_p_output = tf.nn.log_softmax(tf.einsum('nktd,ds->nkts',h_dec,params['V']),-1)
    """compute loss for decoder output under all K components"""
    llik_z = tf.reduce_sum(tf.expand_dims(log_p_z,-1) + tf.reduce_sum(log_p_output*tf.expand_dims(dec_out_,1),-1),-1)
    """marginalize out discrete parameter with log sum exp"""
    lliks = tf.reduce_logsumexp(llik_z,-1)
    return(tf.reduce_sum(lliks))

batch_size = 32

#llik = loglik()*N/batch_size

#mean loss
llik = loglik()/batch_size

#ELBO is just the log-likelihood, no priors
ELBO = llik

tf.summary.scalar("ELBO", ELBO)

optimizer = tf.train.AdamOptimizer(.001)
#gradients, variables = zip(*optimizer.compute_gradients(-ELBO))
#gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
#train_op = optimizer.apply_gradients(zip(gradients, variables))
train_op = optimizer.minimize(-ELBO)
check_op = tf.add_check_numerics_ops()
init=tf.global_variables_initializer()

n_epochs = int(20000*batch_size/N)
chains = 4
posterior = {}
idx = np.arange(N)
for c in range(chains):
    ELBOs = []
    posterior[c] = {}
    np.random.seed(0)
    tf.set_random_seed(0)
    sess = tf.Session()
    sess.run(init)
    for epoch in range(n_epochs):
        n_eps_ = 1
        np.random.shuffle(idx)
        for t in range(int(N/batch_size)):
            print(c,epoch*int(N/batch_size)+t,end=' ')
            batch_idx_ = idx[t*batch_size:(t+1)*batch_size]
            start_time = time.time()
            _, ELBO_, = sess.run([(train_op,check_op), ELBO],feed_dict={batch_idx:batch_idx_})
            duration = time.time() - start_time
            ELBOs.append(ELBO_)
            print(duration,ELBO_)
    for k in var_params.keys():
        posterior[c][k] = (sess.run(var_params[k].mean()),sess.run(var_params[k].stddev()))
    posterior[c]['ELBO'] = ELBOs