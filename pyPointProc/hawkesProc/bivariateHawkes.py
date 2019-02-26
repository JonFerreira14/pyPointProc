import matplotlib.pyplot as plt 
from math import exp, ceil, log
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

def exponentialKernel(alpha, beta, x, y):
	return (alpha*exp(-beta*(x-y)))

def hawkesIntensity(arrivals1, arrivals2, params=[0.3,0.2,0.9,0.9,0.8,0.8,1.0,0.9], granularity=10):
	## Setting params
	mu1, mu2 = params[0], params[1]
	alpha11, alpha12, alpha21, alpha22 = params[2], params[3], params[4], params[5]
	beta1, beta2 = params[6], params[7]
	## Create dataframe to help w/ calc
	T = max(arrivals1[-1], arrivals2[-1])
	hawkesSeries1, hawkesSeries2, timestamps = [], [], []
	
	arrivalData1, arrivalData2 = {'ArrivalTime':arrivals1}, {'ArrivalTime':arrivals2}
	arrivalDF1 = pd.DataFrame(data=arrivalData1)
	arrivalDF2 = pd.DataFrame(data=arrivalData2)

	arrivalDF1 = arrivalDF1.set_index("ArrivalTime", drop=False)
	arrivalDF2 = arrivalDF2.set_index("ArrivalTime", drop=False)
	
	for i in range(0,ceil(T)*granularity+1):
		tempSum1, tempSum2 = 0, 0
		
		## Hawkes series 1
		for j in arrivalDF1.loc[:i/granularity-1/granularity,"ArrivalTime"]:
			tempSum1 += exponentialKernel(alpha11, beta1, i/granularity, j) 
		
		for j in arrivalDF2.loc[:i/granularity-1/granularity,"ArrivalTime"]:
			tempSum1 += exponentialKernel(alpha12, beta1, i/granularity, j) 
		
		## Hawkes series 2
		for j in arrivalDF1.loc[:i/granularity-1/granularity,"ArrivalTime"]:
			tempSum2 += exponentialKernel(alpha21, beta2, i/granularity, j) 
		
		for j in arrivalDF2.loc[:i/granularity-1/granularity,"ArrivalTime"]:
			tempSum2 += exponentialKernel(alpha22, beta2, i/granularity, j) 
		
		hawkesSeries1.append(mu1 + tempSum1)
		hawkesSeries2.append(mu2 + tempSum2)
		timestamps.append(i/granularity)
	

	return hawkesSeries1, hawkesSeries2, timestamps

def thinningFunction():
	return

def compensatorFunction():
	return

def goodnessOfFit():
	return

def logLikelihood():
	return

def fit():
	return

def cumulativeArrivals():
	return
