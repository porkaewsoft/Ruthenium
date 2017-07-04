
"""
An implementation of Recurrent Nueral Network for Sequence Labeling

Running 
THEANO_FLAGS='floatX=float32' python main_SimpleRNN02.py

"""

import utils
import theano
import numpy as np
"""

To Do
0. Embedding Layer
1. Training Step for Sequence Labeling
2. Masking 
3. Regularization / Gradient Clipping [ grad = T.clip(T.grad(cost, [w, w1]), -5.0, 5.0) ]
4. GRU
5. LSTM
6. LAU

"""

import theano.tensor as T
from theano import config
from collections import OrderedDict
from layer import AddLayer, DenseLayer, SimpleRNNLayer, TimeDistributedDenseLayer,EmbeddingLayer
from optimizer import *
from utils_data import load_corpus
from random import shuffle
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Start Logging')

"""CLASS Model_SimpleRNN for Sequence Labeling"""

class Model_SimpleRNN:
    def __init__(self):

        logging.info('Start Building Model...')

        logging.info("config.floatX" + " " + config.floatX)

        TparamD = OrderedDict()

        trg_vocab_size = 20
        src_vocab_size = 20
        hidden_size = 20
        embedding_size = 5

        self.x = T.matrix("x",dtype="int64")
        self.x0 = self.x
        x = self.x
        inp = self.x.dimshuffle((1,0))

        self.layer1 = EmbeddingLayer(inp, vocab_size=src_vocab_size, embedding_size=embedding_size,prefix="Embed01")
        self.forward1 = theano.function([x], self.layer1.output)



        self.layer2 = SimpleRNNLayer(inpSeq=self.layer1.output,mask=None,in_dim=embedding_size,
                                    hidden_dim=hidden_size,bias=1, prefix="RNN01")

        self.forward2 = theano.function([x], self.layer1.output)        
        self.current_output = self.layer2.output[0]


        self.layer3 = TimeDistributedDenseLayer(inpSeq=self.current_output,mask=None,in_dim=hidden_size,out_dim=trg_vocab_size,activation="softmax",prefix="TimeDense01")
        self.forward = theano.function([x],self.layer3.output)

        TparamD = OrderedDict()
        TparamD.update(self.layer1.Tparam)
        TparamD.update(self.layer2.Tparam)
        TparamD.update(self.layer3.Tparam)


        print TparamD
        #Training step 
        """
            Batch of probs  of each time step

            [   [(....),(....)],
                [(....),(....)],
                [(....),(....)]     ] ==> for example, (3 step, 2 batch, 4 classes)

        """
    
        self.y0 = T.matrix("y",dtype="int64")
        self.y = self.y0.dimshuffle((1,0))

        probs = self.layer3.output #layer2.output is already softmax
        y_flat = self.y.flatten() #flatten ids

        y_flat_idx = T.arange(y_flat.shape[0]) * trg_vocab_size + y_flat #shift the ids to match the prob_flat
        cost = -T.log(probs.flatten()[y_flat_idx]) #calcuate log for only picked ids
        cost = cost.reshape([self.y.shape[0], self.y.shape[1]])
        cost = cost.sum(0)

        cost = cost.mean()

        """
        Add Regularize HERE !!
        """

        logging.info("Building Gradient...")
        self.train = theano.function([self.x,self.y],[probs,cost])

        UpdateParams = TparamD

        grads = T.grad(cost, wrt=list(UpdateParams.values()))
        f_grad = theano.function([self.x0, self.y0], grads, name='f_grad')

        lr = T.scalar(name='lr')
        optimizer = adadelta
        self.f_grad_shared, self.f_update = optimizer(lr, UpdateParams, grads,
                                                                   self.x0, self.y0, cost)

        logging.info('Building Model Completed.')


    def predict(self,x):
        print self.forward(x)
        return self.forward(x)
        

    def train_one_batch(self,x,y,lrate=0.01):
        return self.train(x,y)

def batch_generator(srcL,trgL,batch_size=20):
    key = range(len(srcL))
    shuffle(key)
    np_key = np.array(key)
    start = 0
    srcLen= len(srcL)
    while start < srcLen:
        yield srcL[key[start:start+batch_size]],trgL[key[start:start+batch_size]]
        start += batch_size


if __name__ == "__main__":
    """
    Load Corpus -- read src and trg, then convert to ids
    """
    logging.info("Loading Corpus...")
    srcL,trgL = load_corpus("data/src.txt","data/trg.txt","data/src.vocab","data/trg.vocab")
    
    np_srcL = np.array(srcL,dtype="int64")
    np_trgL = np.array(trgL,dtype="int64")


    model = Model_SimpleRNN()  

    i = 0
    lrate = 0.01
    nepoch = 10

    for n in range(nepoch):
        total_loss = 0.0
        for src,trg in batch_generator(np_srcL,np_trgL):
            loss = model.f_grad_shared(src,trg)
            total_loss += loss
            model.f_update(lrate)
            i += 1
        logging.info("Epoch %d %f"%(n,total_loss))

    #model = Model_SimpleRNN()  

    #out = model.predict(np.asarray([[1,2],[2,3],[3,1]],dtype="int64"))

