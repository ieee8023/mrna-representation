
# coding: utf-8

# Batches work better than entire gradient. Full gradients seem to just stall and stop at some non-minimal point. Shaking it up with different learning method can help but this seems like just a hack.
# 
# the network needs to be pretty deep to get reasonable encoding. 
# 
# a full gradient from a loss function that combines pca embedding and the reconstruction loss seems to help the network get near a good local minimum and avoid getting stuck

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T 
import theano
import sys
import os
import time
sys.path.insert(0,"eigengenes")
import load_data
import pickle
import lasagne
import lasagne.layers
import climate
climate.enable_default_logging()


# In[ ]:




# In[ ]:

#sys.argv = ['test.py', 'bthet', '2']
print sys.argv;
#ecoli
#bthet
species = sys.argv[1]
numhidden = int(sys.argv[2])
data =  load_data.load(species, False)[0]
datashape =  data.shape
#data = np.array(data,dtype=theano.config.floatX)
#data = theano.shared(data,name = 'data', borrow = True)
data = np.array(data,dtype=theano.config.floatX)


# In[ ]:




# In[ ]:

from sklearn.decomposition import PCA
#results = PCA(data)
pca = PCA(n_components=numhidden)
pca_embedding = pca.fit_transform(data)
pca_reconstruction = pca.inverse_transform(pca_embedding)


# In[ ]:




# In[ ]:

input_var = T.matrix('inputs')
target_var = T.matrix('targets')

l_in = lasagne.layers.InputLayer(shape=datashape, input_var=input_var)

nonlinearity=lasagne.nonlinearities.rectify

midsize = 200

l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=midsize, 
                                   W=lasagne.init.GlorotUniform(), 
                                   nonlinearity=nonlinearity)

l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=midsize, 
                                   W=lasagne.init.GlorotUniform(), 
                                   nonlinearity=nonlinearity)

l_hidden = lasagne.layers.DenseLayer(l_hid2, num_units=numhidden, 
                                     W=lasagne.init.GlorotUniform(), 
                                     nonlinearity=nonlinearity)

l_hid4 = lasagne.layers.DenseLayer(l_hidden, num_units=midsize, 
                                   W=lasagne.init.GlorotUniform(), 
                                   nonlinearity=nonlinearity)

l_hid5 = lasagne.layers.DenseLayer(l_hid4, num_units=midsize, 
                                   W=lasagne.init.GlorotUniform(), 
                                   nonlinearity=nonlinearity)

l_out = lasagne.layers.DenseLayer( l_hid5, num_units=datashape[1], 
                                  W=lasagne.init.GlorotUniform(), 
                                  nonlinearity=nonlinearity)

model_desc = "{}x{}x{}x{}x{}x{}x{}".format(l_hid1.input_shape[1], 
                                 l_hid2.input_shape[1],  
                                 l_hidden.input_shape[1], 
                                 l_hid4.input_shape[1], 
                                 l_hid5.input_shape[1], 
                                 l_out.input_shape[1],
                                 datashape[1])
print model_desc


# In[ ]:

prediction = lasagne.layers.get_output(l_out)
hidden = lasagne.layers.get_output(l_hidden)
loss = ((prediction - target_var)**2).sum(axis=1).mean()
#+ ((hidden - pca_embedding)**2).sum(axis=1).mean()
hiddenloss = loss + ((hidden - pca_embedding)**2).sum(axis=1).mean()

#define how to make prediction
ae_reconstruct = theano.function(
    inputs=[input_var],
    outputs=lasagne.layers.get_output(l_out),
)

#define how to output embedding
ae_embed = theano.function(
    inputs=[input_var],
    outputs=lasagne.layers.get_output(l_hidden),
)


# In[ ]:

params = lasagne.layers.get_all_params(l_out, trainable=True)


hiddenupdates = lasagne.updates.adadelta(hiddenloss, params, learning_rate=0.01)

pretrain_model = theano.function(
    inputs=[input_var, target_var], 
    outputs=hiddenloss,
    updates=hiddenupdates)

lr = theano.shared(np.array(0.00001, dtype=theano.config.floatX))
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
updates2 = lasagne.updates.adadelta(loss, params, learning_rate=lr)
#updates = lasagne.updates.sgd(loss, params, learning_rate=0.0001)


train_model = theano.function(
    inputs=[input_var, target_var], 
    outputs=loss, 
    updates=updates)

train_model2 = theano.function(
    inputs=[input_var, target_var], 
    outputs=loss, 
    updates=updates2)


# In[ ]:

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]                                              
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# In[ ]:

num_epochs = 2000

print("Starting pretraining...")
# We iterate over epochs:
for epoch in range(num_epochs):
    start_time = time.time()
    train_err = pretrain_model(data,data)
    if epoch % 100 == 0:
        print("Epoch {} of {} took {:.3f}s, loss:\t{:.6f}".format(
            epoch + 1, num_epochs, time.time() - start_time, train_err*1))
        sys.stdout.flush()


# In[ ]:

num_epochs = 20000
numbatches = 10
lr = 0.001
print("Starting batch training...")
# We iterate over epochs:
start_time = time.time()
for epoch in range(num_epochs):
    if epoch % 5000 == 0:
        lr = lr*0.1
        print("Learning rate now:{}".format(lr))
    train_err = 0;
    for train,target in iterate_minibatches(data,data,len(data)/numbatches,True):
        train_err += train_model(train,target)/2
        train_err += train_model(train,target)/2
    if epoch % 50 == 0:
        print("Epoch {} of {} took {:.3f}s, loss:\t{:.6f}".format(
            epoch + 1, num_epochs, time.time() - start_time, train_err*1/numbatches))
        sys.stdout.flush()
        start_time = time.time()


# In[ ]:

## AE reconstruction error
ae_reconstruction = ae_reconstruct(data)
print "ae_reconstruction, species, {}, numhidden, {}, error, {}".format(species, numhidden,(np.sum((ae_reconstruction - data)**2, axis=1)).mean())


# In[ ]:

from sklearn.decomposition import PCA
pca = PCA(n_components=numhidden)
pca_embedding = pca.fit_transform(data)
pca_reconstruction = pca.inverse_transform(pca_embedding)
## PCA reconstruction error
print "pca_reconstruction, species, {}, numhidden, {}, error, {}".format(species, numhidden,(np.sum((pca_reconstruction - data)**2, axis=1)).mean())


# In[ ]:




# In[ ]:




# In[ ]:

pkl_params = lasagne.layers.get_all_param_values(l_out, trainable=True)
pickle.dump(pkl_params, open( species + "_" + model_desc + "_params.p", "wb" ) )


# In[ ]:

#lasagne.layers.set_all_param_values(l_out,pkl_params)


# In[ ]:




# In[ ]:




# In[ ]:

ae_embedding = ae_embed(data)
#ae_embedding


# In[ ]:

#pickle.dump(ae_embedding, open( species + "_" + model_desc + "_ae_embedding.p", "wb" ) )
ae_embedding = pickle.load(open( species + "_" + model_desc + "_ae_embedding.p" ) )


# In[ ]:




# In[ ]:

show = False


# In[ ]:

#show plot of 2d embedding
if show:
    get_ipython().magic(u'matplotlib inline')
    plt.plot(ae_embedding[:,0], ae_embedding[:,1], 'ro')
    plt.show()


# In[ ]:

#show pca of embedding
if show:
    get_ipython().magic(u'matplotlib inline')
    pca = PCA(n_components=2)
    pca_plot = pca.fit_transform(ae_embedding)
    plt.plot(pca_plot[:,0], pca_plot[:,1], 'ro')
    plt.show()


# In[ ]:




# In[ ]:

if show:
    get_ipython().magic(u'matplotlib inline')
    pca = PCA(n_components=2)
    pca_plot = pca.fit_transform(pca_embedding)
    plt.plot(pca_plot[:,0], pca_plot[:,1], 'ro')
    plt.show()


# In[ ]:




# In[ ]:



