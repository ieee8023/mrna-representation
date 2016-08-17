
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T 
import theano
import sys
sys.path.insert(0,"eigengenes")
import load_data
import pickle


# In[ ]:

# uncomment if running in ipython notebook
#sys.argv = ['test.py', 'bthet', '2']

#ecoli
#bthet
species = sys.argv[1]
numhidden = int(sys.argv[2])
data =  load_data.load(species, False)[0]
datashape =  data.shape
data = np.array(data,dtype=theano.config.floatX)
data = theano.shared(data,name = 'data', borrow = True)
#data = np.array(data,dtype=theano.config.floatX)


# In[ ]:




# In[ ]:

from sklearn.decomposition import PCA
pca = PCA(n_components=numhidden)
pca_embedding = pca.fit_transform(data.eval())
pca_reconstruction = pca.inverse_transform(pca_embedding)


# In[ ]:

lr = T.dscalar()
if False:
    #######################################################
    ### Here is a standard linear tied weight autoencoder
    ### This should match PCA reconstruction error
    ### Gradient clipping is used for training
    model_desc = str(datashape[1]) + "x" + str(numhidden) + "x" + str(datashape[1]);

    #define the weights of the model 
    w1 = theano.shared (value = np.random.uniform(low=-1.0, high=1.0 , size=(datashape[1],numhidden)))

    if (False):
        allparams = pickle.load( open( model_desc + "_params.p", "rb" ))
    else:
        allparams = [w1]

    w1 = allparams[0]

    #define the core of the model
    inputx = T.matrix ()
    hidden = T.dot(inputx, theano.gradient.grad_clip(w1,-1,1))
    output = T.dot(hidden, theano.gradient.grad_clip(w1.T,-1,1))
    
    #define loss function
    L = T.mean(T.sum((inputx-output)**2, axis=1))
    
    updates = [(w1, w1 - lr * T.grad(cost = L, wrt = w1))]

elif False:
    #######################################################
    ### Here is a linear autoencoder (without bias though)
    ### Gradient clipping is used for training
    ### The loss function is combined with pca reconstruction in hidden layer
    model_desc = str(datashape[1]) + "x" + str(numhidden) + "x" + str(datashape[1]);

    #define the weights of the model 
    w1 = theano.shared (value = np.random.uniform(low=-1.0, high=1.0 , size=(datashape[1],numhidden)))
    w2 = theano.shared (value = np.random.uniform(low=-1.0, high=1.0 , size=(numhidden, datashape[1])))

    if (False):
        allparams = pickle.load( open( model_desc + "_params.p", "rb" ))
    else:
        allparams = [w1, w2]

    w1, w2 = allparams

    #define the core of the model
    inputx = T.matrix ()
    hidden = T.dot(inputx, theano.gradient.grad_clip(w1,-1,1))
    output = T.dot(hidden, theano.gradient.grad_clip(w2,-1,1))
    
    #define loss function
    L = 0.8*T.mean(T.sum((inputx-output)**2, axis=1)) + 0.2*T.mean(T.sum((pca_embedding-hidden)**2, axis=1))
    
    updates = [(x, x - lr * T.grad(cost = L, wrt = x)) for x in allparams]
    
elif True:
    #######################################################
    ### Here is a deep non-linear autoencoder
    ### The loss function is combined with pca reconstruction in hidden layer
    
    numhidden0 = 4
    numhidden1 = 4
    model_desc = str(datashape[1]) + "x" + str(numhidden0) + "x" + str(numhidden)+ "x" + str(numhidden1) + "x" + str(datashape[1]);

    #define the weights of the model 
    w1 = theano.shared (value = np.random.uniform(low=-1.0, high=1.0 , size=(datashape[1],numhidden0)))
    w2 = theano.shared (value = np.random.uniform(low=-1.0, high=1.0 , size=(numhidden0,numhidden)))
    w3 = theano.shared (value = np.random.uniform(low=-1.0, high=1.0 , size=(numhidden,numhidden1)))
    w4 = theano.shared (value = np.random.uniform(low=-1.0, high=1.0 , size=(numhidden1,datashape[1])))
    
    b1 = theano.shared (value = np.zeros(numhidden0))
    b2 = theano.shared (value = np.zeros(numhidden))
    b3 = theano.shared (value = np.zeros(numhidden1))
    b4 = theano.shared (value = np.zeros(datashape[1]))

    if (False):
        allparams = pickle.load( open( model_desc + "_params.p", "rb" ))
    else:
        allparams = [w1, w2, w3, w4, b1, b2, b3, b4]

    w1, w2, w3, w4, b1, b2, b3, b4 = allparams
    
    inputx = T.matrix ()

    hidden0 = theano.tensor.tanh(T.dot(inputx, theano.gradient.grad_clip(w1,-1,1)) + theano.gradient.grad_clip(b1,-1,1))
    hidden = theano.tensor.tanh(T.dot(hidden0, theano.gradient.grad_clip(w2,-1,1)) + theano.gradient.grad_clip(b2,-1,1))
    hidden1 = theano.tensor.tanh(T.dot(hidden, theano.gradient.grad_clip(w3,-1,1)) + theano.gradient.grad_clip(b3,-1,1))
    output = theano.tensor.tanh(T.dot(hidden1, theano.gradient.grad_clip(w4,-1,1)) + theano.gradient.grad_clip(b4,-1,1))
    
    #define loss function
    L = T.mean(T.sum((inputx-output)**2, axis=1)) + T.mean(T.sum((pca_embedding-hidden)**2, axis=1))

    updates = [(x, x - lr * T.grad(cost = L, wrt = x)) for x in allparams]
        
print("Model shape " + model_desc)


# In[ ]:




# In[ ]:

#define method to train each epoch
train_model = theano.function(
    inputs=[lr],
    outputs=L,
    updates=updates,
    givens = {
        inputx: data
    }
)

#define how to make prediction
ae_reconstruct = theano.function(
    inputs=[inputx],
    outputs=output,
)

#define how to output embedding
ae_embed = theano.function(
    inputs=[inputx],
    outputs=hidden,
)


# In[ ]:

for i in range(3):
    lr = 0.01
    for i in range(2):
        print "Learning rate: {}".format(lr)
        olderror = float("inf")
        for i in range(1000):
            error = train_model(lr)
            if i%50 == 0: 
                print "Epoch: {:>04}, Mean Squared Error: {}".format(i,error)
                if  (error + 0.01) > olderror:
                    print "Inner Loop End"
                    break
                olderror = error
                sys.stdout.flush()
                pickle.dump(allparams, open( model_desc + "_params.p", "wb" ) )
        lr = lr * 0.1


# In[ ]:




# In[ ]:

ae_embedding = ae_embed(data.eval())
#ae_embedding


# In[ ]:




# In[ ]:

show = False


# In[ ]:

if show:
    get_ipython().magic(u'matplotlib inline')
    plt.plot(ae_embedding[:,0], ae_embedding[:,1], 'ro')
    plt.show()


# In[ ]:

from sklearn.decomposition import PCA
#results = PCA(data)
pca = PCA(n_components=numhidden)
pca_embedding = pca.fit_transform(data.eval())
pca_reconstruction = pca.inverse_transform(pca_embedding)


# In[ ]:

if show:
    get_ipython().magic(u'matplotlib inline')
    plt.plot(pca_embedding[:,0], pca_embedding[:,1], 'ro')
    plt.show()


# In[ ]:

## PCA reconstruction error
print "pca_reconstruction, species, {}, numhidden, {}, error, {}".format(species, numhidden,(np.sum((pca_reconstruction - data.eval())**2, axis=1)).mean())


# In[ ]:

## AE reconstruction error
ae_reconstruction = ae_reconstruct(data.eval())
print "ae_reconstruction, species, {}, numhidden, {}, error, {}".format(species, numhidden,(np.sum((ae_reconstruction - data.eval())**2, axis=1)).mean())


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



