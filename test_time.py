"""
Test numpy for time distributed dense layer


dim (a) = (timestep, batch_size, word_embedding_size)

"""

import numpy as np 


a = [	[[1,1,1] , [3,3,3]],
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