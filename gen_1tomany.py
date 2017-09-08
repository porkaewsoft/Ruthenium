import logging
from numpy import random as rand


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Start Logging')

num_token = 10
max_len = 10
max_repeat = 5

def gen_sample():
	token  = rand.random_integers(0,num_token,max_len)
	repeat = rand.random_integers(1,max_repeat,max_len)

	alpha = [chr(65+x) for x in token]
	#print token
	#print alpha
	#print repeat

	#Generate Output
	output = ""
	for i,r in enumerate(repeat):
		for j in range(r):
			output += alpha[i] + " "
	#print output.strip()

	inp = ""
	for n,a in zip(repeat,alpha):
		inp += str(n) + " " + a + " "

	#print inp.strip()
	return inp.strip(),output.strip()


for i in range(0,20000):
	inp,outp = gen_sample()
	print inp + " ||| " + outp
	logging.info('Gen %d'%i)
