
"""
An implementation of Recurrent Nueral Network for Language Modeling

Running
THEANO_FLAGS='floatX=float32' python main_SimpleRNN02.py

"""

import utils
import theano
import numpy as np
import json
"""

To Do
0. Embedding Layer
1. Training Step
2. Masking

"""

import theano.tensor as T
from theano import config
from collections import OrderedDict
from layer import AddLayer, DenseLayer, SimpleRNNLayer, TimeDistributedDenseLayer,EmbeddingLayer,GRULayer
from optimizer import *
from utils_data import load_corpus,load_dictionary_inverse
from random import shuffle
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Start Logging')

"""CLASS Model_SimpleRNN for Sequence Labeling"""

random_seed = 1234
rng = np.random.RandomState(random_seed)

def _dropout_from_layer(layer, p):
    global rng
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class Model_GRULM:
    def __init__(self,load=None):

        logging.info('Start Building Model...')

        logging.info("config.floatX" + " " + config.floatX)

        TparamD = OrderedDict()

        trg_vocab_size = 8
        src_vocab_size = 8
        hidden_size = 60
        embedding_size = 5

        self.hidden_size = hidden_size
        self.x = T.matrix("x",dtype="int64")
        self.x0 = self.x
        x = self.x
        inp = self.x.dimshuffle((1,0))

        self.layer1 = EmbeddingLayer(inp, vocab_size=src_vocab_size, embedding_size=embedding_size,prefix="Embed01")
        self.forward1 = theano.function([x], self.layer1.output)



        self.layer2 = GRULayer(inpSeq=self.layer1.output,mask=None,in_dim=embedding_size,
                                    hidden_dim=hidden_size,bias=1, prefix="RNN01")


        self.forward2 = theano.function([x], self.layer1.output)
        self.current_output = self.layer2.output[0]

        """Dropout at hidden states"""
        self.current_output = _dropout_from_layer(self.current_output,0.2)

        self.layer3 = TimeDistributedDenseLayer(inpSeq=self.current_output,mask=None,in_dim=hidden_size,out_dim=trg_vocab_size,activation="softmax",prefix="TimeDense01")
        self.forward = theano.function([x],self.layer3.output)

        self.one_step_state = np.zeros((hidden_size,),dtype="float32")

        TparamD = OrderedDict()
        TparamD.update(self.layer1.Tparam)
        TparamD.update(self.layer2.Tparam)
        TparamD.update(self.layer3.Tparam)


        print TparamD

        self.TparamD = TparamD

        if load is not None:
            self.load(load)

        isTrain = True

        if isTrain:

            #Training step
            """
                Batch of probs  of each time step
    
                [   [(....),(....)],
                    [(....),(....)],
                    [(....),(....)]     ] ==> for example, (3 step, 2 batch, 4 classes)
    
            """

            self.y0 = T.ones_like(self.x0)
            self.y = T.set_subtensor(self.y0[:,0:-1],self.x0[:,1:])

            self.fy = theano.function([self.x0],self.y)

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
            self.train = theano.function([self.x],[probs,cost])

            UpdateParams = TparamD

            grads = T.grad(cost, wrt=list(UpdateParams.values()))
            f_grad = theano.function([self.x0], grads, name='f_grad')

            lr = T.scalar(name='lr')
            optimizer = adadelta_lm
            self.f_grad_shared, self.f_update = optimizer(lr, UpdateParams, grads,
                                                                       self.x0, cost)

        logging.info('Building Model Completed.')

    def predict(self,x):
        print self.forward(x)
        return self.forward(x)

    """One step computation"""
    def one_step(self,x_id):
        hidden_size = self.hidden_size
        temp = self.layer1.one_step(x_id)
        gru_process = self.layer2.one_step()
        h_new = gru_process(temp,self.one_step_state)
        self.one_step_state = h_new

        out = np.argmax(self.layer3.one_step(h_new))

        return out

    def save(self,filename):
        pool = OrderedDict()
        for key in self.TparamD.keys():
            print type(self.TparamD[key].get_value())
            pool[key] = self.TparamD[key].get_value().tolist()

        fp = open(filename,"w")
        fp.writelines(json.dumps(pool) + "\n")
        fp.close()

    def load(self,filename):
        pool = json.loads(open(filename,"r").read())
        for key in pool:
            self.TparamD[key].set_value(np.array(pool[key],dtype="float32"))

def batch_generator(srcL,trgL,batch_size=2):
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
    srcL,trgL = load_corpus("data/src2","data/trg2","data/src.vocab","data/trg.vocab")
    id2word = load_dictionary_inverse("data/src.vocab")
    np_srcL = np.array(srcL,dtype="int64")
    np_trgL = np.array(trgL,dtype="int64")



    model = Model_GRULM()

    """
    current_input = 0
    for i in range(0,20):
        print id2word[current_input],
        current_input = model.one_step(current_input)
    """
    #model.save("model.json")

    i = 0
    lrate = 0.001
    nepoch = 100

    for n in range(nepoch):
        total_loss = 0.0
        for src,trg in batch_generator(np_srcL,np_trgL):
            print src
            print model.fy(src)
            loss = model.f_grad_shared(src)
            total_loss += loss
            model.f_update(lrate)
            i += 1
        logging.info("Epoch %d %f"%(n,total_loss))

    model.save("model.json")
    #model = Model_SimpleRNN()

    #out = model.predict(np.asarray([[1,2],[2,3],[3,1]],dtype="int64"))
