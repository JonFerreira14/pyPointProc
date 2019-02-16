import matplotlib.pyplot as plt 
from math import exp, sin, ceil, log
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

def exponentialKernel(alpha, beta, x, y):
	
	return (alpha*exp(-beta*(x-y)))

def hawkesIntensity(mu, alpha, beta, arrivals):
	T = arrivals[-1]
	intensity, timestamps = [], []
	stepScale = 10

	df = {"ArrivalTime":arrivals}
	arrivalDF = pd.DataFrame(data=df)
	arrivalDF = arrivalDF.set_index("ArrivalTime", drop=False)

	for i in range(0,ceil(T)*stepScale+1):
		tempSum = 0
		for j in arrivalDF.loc[:i/stepScale-1/stepScale,"ArrivalTime"]:
			tempSum += exponentialKernel(alpha, beta, i/stepScale, j) 
		intensity.append(mu + tempSum)
		timestamps.append(i/stepScale)

	return intensity, timestamps

def thinningFunction(T, mu, alpha, beta):
	epsilon = 10**(-10)
	P = []
	t = 0

	while t<T:
		templist = P
		templist.append(t+epsilon)

		M, disregard = hawkesIntensity(mu,alpha,beta, templist)
		M = M[-1]
		
		E = np.random.exponential(M)
		t = t + round(E,1)
		
		U = np.random.uniform(0,M)
		
		templist[-1]=t
		M, disregard = hawkesIntensity(mu,alpha,beta, templist)
		M = M[-1]

		if t!= P[-1] and t<T and U <= M:
			P.append(t)

	return(P)

def compensatorFunction(mu, alpha, beta, arrivals):
	compensatorValues = []

	df = {"ArrivalTime":arrivals}
	arrivalDF = pd.DataFrame(data=df)
	arrivalDF = arrivalDF.set_index("ArrivalTime", drop=False)


	for row in arrivalDF.itertuples():
		tempSum = 0
		
		for j in arrivalDF.loc[:row[1],"ArrivalTime"]:
			if j == row[1] or row[1] == 0:
				continue
			tempSum += exponentialKernel(1, beta, row[1], j)-1


		compensatorValues.append(mu*row[1] - (alpha/beta)*tempSum)

	return compensatorValues

def goodnessOfFit(compensatorValues):
	diffs = []
	for i in range(0, len(compensatorValues)):
		if i == 0:
			diffs.append(compensatorValues[i])
		else:
			diffs.append(compensatorValues[i]-compensatorValues[i-1])
	
	rval = stats.probplot(diffs, dist='expon', plot=None)#plt)
	rsquared = rval[1][2]**2

	#plt.show()

	return rsquared

def logLikelihood(mu, alpha, beta, arrivals):
	
	T = arrivals[-1]
	
	#find firstSum
	firstSum = 0
	for i in arrivals:
		timeDifference = T - i
		timeExponential = exp(-beta*timeDifference)-1
		firstSum += alpha/beta * timeExponential
	
	
	#find rs
	R = np.zeros(len(arrivals))

	for i in range(1,len(R)):
		R[i] += exp(-beta*(arrivals[i] - arrivals[i-1]))*(1+R[i-1])

	
	secondSum = 0
	for i in R:
		secondSum =+ log(mu + alpha*i)
	   
	logLikelihood = -(-mu*T + firstSum + secondSum);
	return logLikelihood

def fit(mu, alpha, beta, arrivals):
	
	def myFitFunc(x):
		p1 = abs(x[0])
		p2 = abs(x[1])
		p3 = abs(x[2])+abs(x[1])
		valToMin = logLikelihood(p1,p2,p3,arrivals)
		print(valToMin)
		return valToMin

	
	x0 = [mu, alpha, beta]
	bnds = ((0,np.inf),(0.001, np.inf),(0.001, np.inf))
	mini = minimize(myFitFunc, x0, method='Powell', options={'maxiter':10000, 'disp':True, 'xtol':1**-20,'ftol':1**-20})#, bounds=bnds)
	return abs(mini.x[0]), abs(mini.x[1]), (abs(mini.x[1])+abs(mini.x[2]))


def arrivalFrequency(arrivals, bucketSize):
	return

def cummulativeArrivals(arrivals):
	return



if __name__ == "__main__":
	alpha = 0.6
	beta = 0.9
	mu = 0.3

	simulatedArrivals = thinningFunction(100, alpha, beta, mu)
	#print(simulatedArrivals)	

	intensity, timestamps = hawkesIntensity(mu, alpha, beta, simulatedArrivals)

	compensatorValues = compensatorFunction(mu, alpha, beta, simulatedArrivals)
	print(goodnessOfFit(compensatorValues))

	plt.plot(timestamps, intensity)
	#plt.plot(simulatedArrivals, compensatorValues)
	plt.show()


	#optimize
	print('optimizing...')
	alpha, beta, mu = fit(alpha, beta, mu, simulatedArrivals)
	print("opt vals are",alpha, beta, mu)

	intensity, timestamps = hawkesIntensity(mu, alpha, beta, simulatedArrivals)

	compensatorValues = compensatorFunction(mu, alpha, beta, simulatedArrivals)
	print(goodnessOfFit(compensatorValues))

	plt.plot(timestamps, intensity)
	#plt.plot(simulatedArrivals, compensatorValues)
	plt.show()