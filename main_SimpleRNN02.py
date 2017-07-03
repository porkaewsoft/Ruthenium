
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

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Start Logging')

"""CLASS Model_SimpleRNN for Sequence Labeling"""

class Model_SimpleRNN:
    def __init__(self):

        logging.info('Start Building Model...')

        x = T.matrix("x",dtype="int64")

        self.layer1 = EmbeddingLayer(x, vocab_size=10, embedding_size=5)
        self.forward = theano.function([x], self.layer1.output)

        logging.info('Building Model Completed.')


    def predict(self,x):
        print self.forward(x)
        return self.forward(x)
        

    def train(self):
        lrate = 0.01
        nepoch = 100000
        

if __name__ == "__main__":
    model = Model_SimpleRNN()  

    out = model.predict(np.asarray([[1,2],[2,3],[3,1]],dtype="int64"))
