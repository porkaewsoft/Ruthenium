import theano
import numpy as np
import theano.tensor as T

from theano import config
from collections import OrderedDict

# Initialize Theano shared variables according to the initial parameters
def Init_theano_params(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def Ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def Norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = Ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(config.floatX)

def Norm_Vector(dim,scale=0.01):
    W = scale * np.random.randn(dim)
    return W.astype(config.floatX)
