
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
from layer import AddLayer, DenseLayer, SimpleRNNLayer, TimeDistributedDenseLayer
from optimizer import *

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Start Logging')

"""CLASS Model_SimpleRNN for Sequence Labeling"""

class Model_SimpleRNN:
    def __init__(self):

        logging.info('Start Building Model...')

        in_dim = 2
        out_dim = 5

        self.x = T.tensor3("x",dtype=config.floatX)


        self.layer1 = SimpleRNNLayer(inpSeq=self.x,mask=None,in_dim=2,
                                    hidden_dim=5,bias=1)
        
        self.forward = theano.function([self.x],self.layer1.output)
        self.current_output = self.layer1.output[0]


        self.layer2 = TimeDistributedDenseLayer(inpSeq=self.current_output,mask=None,in_dim=5,out_dim=5,activation="softmax")
        self.forward2 = theano.function([self.x],self.layer2.output)


        #Training step 
        """
            Batch of probs  of each time step

            [   [(....),(....)],
                [(....),(....)],
                [(....),(....)]     ] ==> for example, (3 step, 2 batch, 4 classes)

        """
        probs = self.layer2.output #layer2.output is already softmax
    
        self.y = T.matrix("y",dtype="int64")

        y_flat = self.y.flatten() #flatten ids

        y_flat_idx = T.arange(y_flat.shape[0]) * out_dim + y_flat #shift the ids to match the prob_flat
        cost = -T.log(probs.flatten()[y_flat_idx]) #calcuate log for only picked ids
        cost = cost.reshape([self.y.shape[0], self.y.shape[1]])
        cost = cost.sum(0)

        cost = cost.mean()

        """
        Add Regularize HERE !!
        """

        self.train = theano.function([self.x,self.y],[probs,cost,y_flat_idx])

        logging.info('Building Model Completed.')


    def predict(self,x):
        print self.forward(x)
        return self.forward2(x)
        

    def train(self):
        lrate = 0.01
        nepoch = 100000
        

if __name__ == "__main__":
    model = Model_SimpleRNN()
    
    x = np.asarray([[[1.0,1.0],[1.0,1.0]],[[1.0,1.0],[1.0,1.0]],[[1.0,1.0],[1.0,1.0]]],dtype="float32")
    y = np.asarray([[1,1],[1,1],[2,2]],dtype="int64")
    #out = model.predict(np.asarray([[[1.0,1.0],[1.0,1.0]],[[1.0,1.0],[1.0,1.0]],[[1.0,1.0],[1.0,1.0]]],dtype="float32"))

    out = model.train(x,y)

    print out[0]
    print out[1]
    print out[2]