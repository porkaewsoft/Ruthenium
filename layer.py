from  utils import Norm_weight,Ortho_weight,Init_theano_params,Norm_Vector
import theano
import numpy as np
import theano.tensor as T

from theano import config
from collections import OrderedDict

class AddLayer:
    def __init__(self,inps,prefix="",suffix=""):
        """
        Simple Layer for
            output = x + y
        """
        self.x = inps[0]
        self.y = inps[1]
        self.Tparam = OrderedDict()

        self.__build__()

    def __build__(self):
        self.forward = theano.function([self.x,self.y],self.x+self.y)

    def load_Tparam(self):
        return self.Tparam

class DenseLayer:
    def __init__(self,inp,in_dim=None,out_dim=None,bias=None,initializer="normal",scale=1.0,activation="tanh",prefix="",suffix=""):
        """
        Simple Dense Layer :
            output = inps.dot(W) + b #if bias is not None
            output = inps.dot(W)     #if bias is None
        """
        self.prefix = prefix
        self.suffix = suffix
        self.bias = bias
        self.activation = activation
        self.inp = inp

        NP_param = OrderedDict()

        if initializer == "normal":
            NP_param[prefix + "W" + suffix] = Norm_weight(in_dim,out_dim,scale=scale)
        elif initializer == "ortho":
            NP_param[prefix + "W" + suffix] = Ortho_weight(in_dim)
        else:
            raise NotImplementedError

        if bias is not None:
            NP_param[prefix + "b" + suffix] = Norm_Vector(out_dim,scale=scale)

        self.Tparam = Init_theano_params(NP_param)
        self.__build__()


    def __build__(self):
        W = self.Tparam[self.prefix + "W" + self.suffix]
        x = self.inp

        if self.bias is not None:
            b = self.Tparam[self.prefix + "b" + self.suffix]
            preact = T.dot(x,W) + b
        else:
            preact = T.dot(x,W)

        if self.activation == "tanh":
            out = T.tanh(preact)
        elif self.activation == "sigmoid":
            out = T.nnet.sigmoid(preact)
        elif self.activation == "relu":
            out = T.nnet.relu(preact)

        self.forward = theano.function([self.inp],out)
        self.output = out

    def load_Tparam(self):
        return self.Tparam


class TimeDistributedDenseLayer():

    """
    TimeDistributedDenseLayer is for apply FF layer to sequence (support batch).

    input is 3d tensor, first dim is timestep, second is batch, third is the element value of vector

    Basic Idea

    import numpy as np 


    a = [   [[1,1,1] , [3,3,3]],
            [[1,1,1] , [3,3,3]],
            [[1,1,1] , [3,3,3]]  ]


    w = [ [2,2,2,2],
          [1,1,1,1],
          [0,0,0,0] ]

    num_a = np.array(a,dtype="float32")
    num_w = np.array(w,dtype="float32")

    reshape_a = num_a.reshape((-1,3))

    print reshape_a

    b = reshape_a.dot(num_w)

    print b

    reshape_b = b.reshape((-1,2,3))

    print reshape_b

    """
    def __init__(self,inpSeq,mask=None,in_dim=None,out_dim=None,activation="softmax",bias=None,dropout=0.5,scale=1.0,prefix="",suffix=""):
        self.prefix = prefix
        self.suffix = suffix
        self.bias = bias
        self.activation = activation
        self.inpS = inpSeq
        self.out_dim = out_dim

        NP_param = OrderedDict()
        NP_param[prefix + "W" + suffix] = Norm_weight(in_dim,out_dim,scale=scale)

        """ToDo Implement bias"""

        self.Tparam = Init_theano_params(NP_param)
        self.__build__()

    def __build__(self):
        prefix = self.prefix
        suffix = self.suffix
        n_timestep = self.inpS.shape[0]
        batch_size = self.inpS.shape[1]
        word_embbed_size = self.inpS.shape[2]
        out_dim = self.out_dim

        inpS = self.inpS
        W = self.Tparam[prefix + "W" + suffix]
        

        seq = inpS.reshape((-1,word_embbed_size))
        
        if self.activation == "softmax":
            output = T.nnet.softmax(T.dot(seq,W))
        elif self.activation == "tanh":
            output = T.tanh(T.dot(seq,W))
        elif self.activation == "relu":
            output = T.nnet.relu(T.dot(seq,W))

        output = output.reshape((n_timestep,batch_size,out_dim))

        #self.forward is used to test only do not use in Model implementation
        self.forward = theano.function([self.inpS],output)
        self.output = output


class SimpleRNNLayer():
    """
        x is input
        h_ is hidden state
        b is bias
        g is activation function

        g(Uh_ + Wx + b)

        U is Matrix(hdim,hdim)
        W is Matrix(in_dim,hdim)

        Note: For batch processing

        Uh_ will be T.dot(h_(batch),U);
        same as Wx

        TODO:
            Implement dropout

    """
    def __init__(self,inpSeq,mask=None,in_dim=None,hidden_dim=100,bias=None,init_state=None,dropout=0.5,activation="sigmoid",prefix="",suffix=""):

        self.prefix = prefix
        self.suffix = suffix
        self.bias = bias
        self.activation = activation
        self.inp = inpSeq
        self.mask = mask
        self.init_state = init_state
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        NP_param = OrderedDict()

        NP_param[prefix + "W" + suffix] = Norm_weight(self.in_dim,hidden_dim)
        NP_param[prefix + "U" + suffix] = Ortho_weight(hidden_dim)

        print NP_param[prefix + "U" + suffix]

        if bias is not None:
            NP_param[prefix + "b" + suffix] = np.zeros((hidden_dim,)).astype("float32")        
        self.Tparam = Init_theano_params(NP_param)
        self.__build__()        

    def __build__(self):
        prefix = self.prefix
        suffix = self.suffix
        n_timestep = self.inp.shape[0]
        
        if self.inp.ndim == 3:
            batch_size = self.inp.shape[1] #row is time step, col is batch id, depth is value of vector.
            dim_inp    = self.inp.shape[2] #the size of input vector 
        else:
            batch_size = 1
            dim_inp    = self.inp.shape[1] #the size of input vector

        hdim = self.hidden_dim

        if self.init_state == None:
            self.init_state = T.zeros((batch_size,hdim))
        
        init_state = self.init_state

        if self.mask is None:
            mask = T.ones((n_timestep, 1))
        else:
            mask = self.mask

        """
        _step is called by theano.scan
            sequence = [m_,inp_]
            output_infos = [self.init_state]

        """
        U = self.Tparam[prefix + "U" + suffix]
        W = self.Tparam[prefix + "W" + suffix]
        
        if self.bias is not None:
            b = self.Tparam[prefix + "b" + suffix]
        
        def _step(m_,inp_,h_,U,W,b=None):
            preact = T.dot(inp_,W) + T.dot(h_,U)
            if self.bias is not None:
                preact += b
            h = preact
            return h

        seqs = [mask,self.inp]
        nonseqs = [U,W]

        if self.bias is not None:
            nonseqs += [b]

        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_state,
                                non_sequences=nonseqs,
                                n_steps=n_timestep,
                                truncate_gradient=-1,
                                strict=False)

        out = [rval]

        self.forward = theano.function([self.inp],out)
        self.output = out

    def load_Tparam(self):
        return self.Tparam