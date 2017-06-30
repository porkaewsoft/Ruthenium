import utils
import theano
import numpy as np
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
        self.x = T.tensor3("x",dtype="float32")

        self.layer1 = SimpleRNNLayer(inpSeq=self.x,mask=None,in_dim=2,
                                    hidden_dim=5,bias=1)
        
        self.forward = theano.function([self.x],self.layer1.output)
        self.current_output = self.layer1.output[0]


        self.layer2 = TimeDistributedDenseLayer(inpSeq=self.current_output,mask=None,in_dim=5,out_dim=5)
        self.forward2 = theano.function([self.x],self.layer2.output)


        logging.info('Building Model Completed.')


    def predict(self,x):
        print self.forward(x)
        return self.forward2(x)
        

    def train(self):
        lrate = 0.01
        nepoch = 100000
        

if __name__ == "__main__":
    model = Model_SimpleRNN()
    out = model.predict(np.asarray([[[1.0,1.0],[1.0,1.0]],[[1.0,1.0],[1.0,1.0]],[[1.0,1.0],[1.0,1.0]]],dtype="float32"))

    print out