import logging
from numpy import random as rand

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Start Logging')

dictA = {"A" : "1" ,"B" : "2" ,"C" : "3" , "D" : "4" , "E" : "5" }
key = dictA.keys()

for i in range(0,1000):
	x = rand.random_integers(0,4,10)
	print " ".join([key[item] for item in x]),
	print "|||",
	y = [dictA[key[item]] for item in x]
	print " ".join(y)

loggin.info("Finish !")



