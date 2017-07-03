
"""
An implementation of Recurrent Nueral Network for Sequence Labeling
"""

import utils
import theano
import numpy as np
"""

To Do
0. Embedding Layer
1. Training Step for Sequence Labeling
2. Masking
3. Regularization
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

        trg_vocab_size = 20
        src_vocab_size = 20
        hidden_size = 20
        embedding_size = 5

        self.x = T.matrix("x",dtype="int64")
        x = self.x
        inp = self.x.dimshuffle((1,0))

        self.layer1 = EmbeddingLayer(inp, vocab_size=src_vocab_size, embedding_size=embedding_size)
        self.forward1 = theano.function([x], self.layer1.output)

        self.layer2 = SimpleRNNLayer(inpSeq=self.layer1.output,mask=None,in_dim=embedding_size,
                                    hidden_dim=hidden_size,bias=1)

        self.forward2 = theano.function([x], self.layer1.output)        
        self.current_output = self.layer2.output[0]


        self.layer3 = TimeDistributedDenseLayer(inpSeq=self.current_output,mask=None,in_dim=hidden_size,out_dim=trg_vocab_size,activation="softmax")
        self.forward = theano.function([x],self.layer3.output)


        #Training step 
        """
            Batch of probs  of each time step

            [   [(....),(....)],
                [(....),(....)],
                [(....),(....)]     ] ==> for example, (3 step, 2 batch, 4 classes)

        """
    
        self.y = T.matrix("y",dtype="int64")
        self.y = self.y.dimshuffle((1,0))

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

        self.train = theano.function([self.x,self.y],[probs,cost])


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

    logging.info("Loading Corpus...")
    srcL,trgL = load_corpus("data/src.txt","data/trg.txt","data/src.vocab","data/trg.vocab")
    
    np_srcL = np.array(srcL,dtype="int64")
    np_trgL = np.array(trgL,dtype="int64")


    model = Model_SimpleRNN()  

    i = 0

    for src,trg in batch_generator(np_srcL,np_trgL):
        out = model.train(src,trg)
        print i,out[1]
        i += 1

    #model = Model_SimpleRNN()  

    #out = model.predict(np.asarray([[1,2],[2,3],[3,1]],dtype="int64"))
