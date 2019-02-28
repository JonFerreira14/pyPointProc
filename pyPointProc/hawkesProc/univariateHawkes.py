import matplotlib.pyplot as plt 
from math import exp, sin, ceil, log
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

def exponentialKernel(alpha, beta, x, y):
	return (alpha*exp(-beta*(x-y)))

def hawkesIntensity(mu, alpha, beta, arrivals, stepScale=10, plot=False):
	T = arrivals[-1]
	intensity, timestamps = [], []

	df = {"ArrivalTime":arrivals}
	arrivalDF = pd.DataFrame(data=df)
	arrivalDF = arrivalDF.set_index("ArrivalTime", drop=False)

	for i in range(0,ceil(T)*stepScale+1):
		tempSum = 0
		for j in arrivalDF.loc[:i/stepScale-1/stepScale,"ArrivalTime"]:
			tempSum += exponentialKernel(alpha, beta, i/stepScale, j) 
		intensity.append(mu + tempSum)
		timestamps.append(i/stepScale)

	if plot == True:
		plt.plot(timestamps, intensity)
		plt.show()

	return intensity, timestamps

def thinningFunction(T, mu, alpha, beta, rounding=1):
	epsilon = 10**(-10)
	P = []
	t = 0

	while t<T:
		templist = P
		templist.append(t+epsilon)

		M, disregard = hawkesIntensity(mu,alpha,beta, templist)
		M = M[-1]
		
		E = np.random.exponential(M)
		t = t + round(E, rounding)
		
		U = np.random.uniform(0,M)
		
		templist[-1]=t
		M, disregard = hawkesIntensity(mu,alpha,beta, templist)
		M = M[-1]

		if t!= P[-1] and t<T and U <= M:
			P.append(t)

	return(P)

def compensatorFunction(mu, alpha, beta, arrivals, plot=False):
	compensatorValues = []

	df = {"ArrivalTime":arrivals}
	arrivalDF = pd.DataFrame(data=df)
	arrivalDF = arrivalDF.set_index("ArrivalTime", drop=False)


	for row in arrivalDF.itertuples():
		tempSum = 0
		
		for j in arrivalDF.loc[:row[1],"ArrivalTime"]:
			if row[1] == 0:
				continue
			tempSum += exponentialKernel(1, beta, row[1], j)-1


		compensatorValues.append(mu*row[1] - (alpha/beta)*tempSum)

	if plot == True:
		plt.plot(arrivals, compensatorValues)
		plt.show()

	return compensatorValues

def goodnessOfFit(compensatorValues, Plot=False):
	diffs = []
	for i in range(0, len(compensatorValues)):
		if i == 0:
			diffs.append(compensatorValues[i])
		else:
			diffs.append(compensatorValues[i]-compensatorValues[i-1])
	
	if Plot == True:
		Plot = plt
	else: 
		Plot = None

	rval = stats.probplot(diffs, dist='expon', plot=Plot)
	rsquared = rval[1][2]**2

	if Plot == plt:
		plt.show()

	return rsquared

def logLikelihood(mu, alpha, beta, arrivals):
	T = arrivals[-1]
	
	firstSum = 0
	for i in arrivals:
		timeDifference = T - i
		timeExponential = exp(-beta*timeDifference)-1
		firstSum += alpha/beta * timeExponential	

	R = np.zeros(len(arrivals))

	for i in range(1,len(R)):
		R[i] += exp(-beta*(arrivals[i] - arrivals[i-1]))*(1+R[i-1])
	
	secondSum = 0
	for i in R:
		secondSum += log(mu + alpha*i)
	   
	
	logLikelihood = -(-mu*T + firstSum + secondSum);
	return logLikelihood

def fit(mu, alpha, beta, arrivals):
	
	def myFitFunc(x):
		p1 = abs(x[0])
		p2 = max(min(abs(x[1]), abs(x[2]), 2),.0001)
		p3 = max(min(abs(x[2]),2),.0001)
		return logLikelihood(p1,p2,p3,arrivals)
	

	x0 = [mu, alpha, beta]
	runLimit = 10
	iteration = 0
	bestFuncVal = 10000000000
	bestx0 = 0

	while iteration < runLimit:
		
		mini = minimize(myFitFunc, x0, method='Nelder-Mead', options={'maxiter':10000, 'maxfev':10000, 'disp':False})#, 'xatol':10**-6,'fatol':10*-6})
		
		currentFuncVal = mini.fun
		currentx0 = [abs(mini.x[0]), max(min(abs(mini.x[1]), abs(mini.x[2]), 2),.0001), max(min(abs(mini.x[2]),2),.0001)]

		if currentFuncVal < bestFuncVal:
			bestFuncVal = currentFuncVal
			bestx0 = currentx0


		x0 = np.random.rand(3)
		iteration+=1
	
	if bestx0 == 0:
		print("error: no solution found")
		return mu, alpha, beta

	return bestx0[0], bestx0[1], bestx0[2]

def cumulativeArrivals(arrivals, plot=False):
	nCount = np.ones_like(arrivals)

	for i in range(1,len(nCount)):
		nCount[i] = nCount[i] + nCount[i-1]


	if plot == True:
		plt.plot(arrivals, nCount)
		plt.show()

	return nCount

