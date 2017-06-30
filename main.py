import utils
import theano
import numpy as np
import theano.tensor as T
from theano import config
from collections import OrderedDict
from layer import AddLayer, DenseLayer
from optimizer import *

class Model_XOR:
    def __init__(self):

        self.x = T.dvector("x")
        self.y = T.dvector("y")

        self.layer1 = DenseLayer(self.x,in_dim=2,out_dim=3,bias=1,activation="sigmoid",prefix="XOR_")
        self.layer2 = DenseLayer(self.layer1.output,in_dim=3,out_dim=2,bias=1,activation="sigmoid",prefix="XOR_2")

        self.forward = theano.function([self.x],self.layer2.output)

        cost = T.sum((self.layer2.output - self.y)**2)

        self.f_cost = theano.function([self.x,self.y],cost)

        Tparam = OrderedDict()
        Tparam.update(dict(self.layer2.load_Tparam()))
        Tparam.update(dict(self.layer1.load_Tparam()))

        grads = T.grad(cost, wrt=list(Tparam.values()))

        lr = T.scalar(name='lr')
        optimizer = adadelta
        self.f_grad_shared, self.f_update = optimizer(lr, Tparam, grads,
                                            self.x, self.y, cost)

    def train(self):
        lrate = 0.01
        nepoch = 100000
        for i in range(0,nepoch):
            print i
            loss  = self.f_grad_shared(np.array([0.1,0.1],dtype=config.floatX),
                                       np.array([0.9,0.1],dtype=config.floatX))
            self.f_update(lrate)

            loss  += self.f_grad_shared(np.array([0.1,0.9],dtype=config.floatX),
                                       np.array([0.0,0.9],dtype=config.floatX))
            self.f_update(lrate)

            loss += self.f_grad_shared(np.array([0.9,0.1],dtype=config.floatX),
                                       np.array([0.1,0.9],dtype=config.floatX))
            self.f_update(lrate)

            loss += self.f_grad_shared(np.array([0.9,0.9],dtype=config.floatX),
                                       np.array([0.9,0.1],dtype=config.floatX))
            self.f_update(lrate)
            print loss

        print self.forward(np.array([0.0,0.0],dtype=config.floatX))
        print self.forward(np.array([0.0,1.0],dtype=config.floatX))
        print self.forward(np.array([1.0,0.0],dtype=config.floatX))
        print self.forward(np.array([1.0,1.0],dtype=config.floatX))

if __name__ == "__main__":
    model = Model_XOR()
    model.train()






def old_main():
    x = T.fvector("x")
    y = T.fvector("y")

    z = AddLayer([x,y])

    output = z.forward(np.asarray([1.,1.],dtype="float32"),
               np.asarray([1.,1.],dtype="float32"))

    z1 = DenseLayer(x,in_dim=2,out_dim=5,bias=1,activation="sigmoid",prefix="D1_")

    output = z1.forward(np.asarray([1.,1.],dtype="float32"))

    print output
