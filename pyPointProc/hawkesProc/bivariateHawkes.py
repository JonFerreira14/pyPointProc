import matplotlib.pyplot as plt 
from math import exp, ceil, log
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

def exponentialKernel(alpha, beta, x, y):
	return (alpha*exp(-beta*(x-y)))

def hawkesIntensity(arrivals1, arrivals2, params=[0.3,0.1,0.6,0.9,0.2,0.5,1.2,1.0], granularity=10):
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

def compensatorFunction(arrivals1, arrivals2, params=[0.3,0.1,0.6,0.9,0.2,0.5,1.2,1.0]):
	## Setting params
	mu1, mu2 = params[0], params[1]
	alpha11, alpha12, alpha21, alpha22 = params[2], params[3], params[4], params[5]
	beta1, beta2 = params[6], params[7]

	compensator1, compensator2 = [], []

	df1, df2 = {"ArrivalTime":arrivals1}, {"ArrivalTime":arrivals2}
	
	arrivalDF1 = pd.DataFrame(data=df1)
	arrivalDF2 = pd.DataFrame(data=df2)

	arrivalDF1labels = arrivalDF1.copy()
	arrivalDF2labels = arrivalDF2.copy()
	arrivalDF1labels = arrivalDF1labels.set_index("ArrivalTime", drop=False)
	arrivalDF2labels = arrivalDF2labels.set_index("ArrivalTime", drop=False)

	# Comp 1
	for row in arrivalDF1.itertuples():
		tempSum = 0
		
		for j in arrivalDF1labels.loc[:row[1],"ArrivalTime"]:
			if row[0] == 0:
				continue
			tempSum += (alpha11/beta1)*(exponentialKernel(1, beta1, row[1], j)-1)
		
		for j in arrivalDF2labels.loc[:row[1],"ArrivalTime"]:
			if row[0] == 0:
				continue
			tempSum += (alpha12/beta1)*(exponentialKernel(1, beta1, row[1], j)-1)

		compensator1.append(mu1*row[1] - tempSum)
	
	# Comp 2
	for row in arrivalDF2.itertuples():
		tempSum = 0
		
		for j in arrivalDF1labels.loc[:row[1],"ArrivalTime"]:
			if row[0] == 0:
				continue
			tempSum += (alpha21/beta2)*(exponentialKernel(1, beta2, row[1], j)-1)
		
		for j in arrivalDF2labels.loc[:row[1],"ArrivalTime"]:
			if row[0] == 0:
				continue
			tempSum += (alpha22/beta2)*(exponentialKernel(1, beta2, row[1], j)-1)

		compensator2.append(mu2*row[0] - tempSum)

	return compensator1, compensator2

def goodnessOfFit(compensatorValues, Plot=True):
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

def logLikelihood(arrivals1, arrivals2, params=[0.3,0.1,0.6,0.9,0.2,0.5,1.2,1.0]):
	## Setting params
	mu1, mu2 = params[0], params[1]
	alpha11, alpha12, alpha21, alpha22 = params[2], params[3], params[4], params[5]
	beta1, beta2 = params[6], params[7]

	T = max(arrivals1[-1], arrivals2[-1])

	## LL = LL(1) + LL(2)
	## LL(1) = -mu1*T + comp val 1+
	
	## Calc compensator sums
	sum11 = 0
	for i in arrivals1:
		sum11 += (1 - exponentialKernel(1, beta1, T, i))
	sum11 = (alpha11/beta1)*sum11

	sum12 = 0
	for i in arrivals2:
		sum12 += (1 - exponentialKernel(1, beta1, T, i))
	sum12 = (alpha12/beta1)*sum12

	## Calc recursive sum
	sum13 = 0

	##Calc R11 and R12
	R11, R12 = np.zeros(len(arrivals1)), np.zeros(len(arrivals1))

	for i in range(1, len(R11)):
		R11[i] = exponentialKernel(1, beta1, arrivals1[i], arrivals1[i-1])*(1+R11[i-1])

	for i in range(1, len(R11)):
		tempSum1 = 0

		for j in arrivals2:
			if j >= arrivals1[i-1] and j < arrivals1[i]:
				tempSum1 += exponentialKernel(1, beta1, arrivals1[i], j)

		R12[i] = exponentialKernel(1, beta1, arrivals1[i], arrivals1[i-1])*(R12[i-1])+tempSum1

	for i in range(1,len(R11)):
		sum13 += log(mu1 + alpha11*R11[i] + alpha12*R12[i])

	logLikelihood1 = -mu1*T - sum11 - sum12 + sum13


	## LL(2)
	## Calc compensator sums
	sum21 = 0
	for i in arrivals1:
		sum21 += (1 - exponentialKernel(1, beta2, T, i))
	sum21 = (alpha21/beta2)*sum21

	sum22 = 0
	for i in arrivals2:
		sum22 += (1 - exponentialKernel(1, beta2, T, i))
	sum22 = (alpha22/beta2)*sum22

	## Calc recursive sum
	sum23 = 0

	##Calc R21 and R22
	R21, R22 = np.zeros(len(arrivals2)), np.zeros(len(arrivals2))

	for i in range(1, len(R22)):
		R22[i] = exponentialKernel(1, beta2, arrivals2[i], arrivals2[i-1])*(1+R22[i-1])

	for i in range(1, len(R22)):
		tempSum2 = 0

		for j in arrivals1:
			if j >= arrivals2[i-1] and j < arrivals2[i]:
				tempSum2 += exponentialKernel(1, beta2, arrivals2[i], j)

		R21[i] = exponentialKernel(1, beta1, arrivals1[i], arrivals1[i-1])*(R21[i-1])+tempSum2

	for i in range(1,len(R22)):
		sum23 += log(mu2 + alpha21*R21[i] + alpha22*R22[i])

	logLikelihood1 = -mu1*T - sum11 - sum12 + sum13
	logLikelihood2 = -mu2*T - sum21 - sum22 + sum23

	logLikelihood = logLikelihood1 + logLikelihood2
	return -logLikelihood

def fit(arrivals1, arrivals2, params=[0.3,0.1,0.6,0.9,0.2,0.5,1.2,1.0]):
	
	def myFitFunc(x):
		#find a beter way of doing this
		x1 = min(max(x[0], 0.0001), 2)
		x2 = min(max(x[1], 0.0001), 2)
		x3 = min(max(x[2], 0.0001), 2)
		x4 = min(max(x[3], 0.0001), 2)
		x5 = min(max(x[4], 0.0001), 2)
		x6 = min(max(x[5], 0.0001), 2)
		x7 = min(max(x[6], 0.0001), 2)
		x8 = min(max(x[7], 0.0001), 2)

		tList = []
		tList.append(x1)
		tList.append(x2)
		tList.append(x3)
		tList.append(x4)
		tList.append(x5)
		tList.append(x6)
		tList.append(x7)
		tList.append(x8)

		return logLikelihood(arrivals1, arrivals2, tList)
	
	x0 = params
	runLimit = 50
	iteration = 0
	bestFuncVal = 10000000000
	bestx0 = []

	while iteration < runLimit:
		
		mini = minimize(myFitFunc, x0, method='Nelder-Mead', options={'disp':False})#,'maxiter':10000, 'xtol':10**-20,'ftol':10**-20})
		
		currentFuncVal = mini.fun
		currentx0 = mini.x

		if currentFuncVal < bestFuncVal:
			bestFuncVal = currentFuncVal
			bestx0 = currentx0


		x0 = np.random.rand(8)
		iteration+=1
	
	if not bestx0.tolist():
		print("error: no solution found")
		return params

	#find a beter way of doing this
	x1 = min(max(bestx0[0], 0.0001), 2)
	x2 = min(max(bestx0[1], 0.0001), 2)
	x3 = min(max(bestx0[2], 0.0001), 2)
	x4 = min(max(bestx0[3], 0.0001), 2)
	x5 = min(max(bestx0[4], 0.0001), 2)
	x6 = min(max(bestx0[5], 0.0001), 2)
	x7 = min(max(bestx0[6], 0.0001), 2)
	x8 = min(max(bestx0[7], 0.0001), 2)

	tList = []
	tList.append(x1)
	tList.append(x2)
	tList.append(x3)
	tList.append(x4)
	tList.append(x5)
	tList.append(x6)
	tList.append(x7)
	tList.append(x8)
	return tList

def cumulativeArrivals(arrivals1, arrivals2):
	nCount1, nCount2 = np.ones_like(arrivals1), np.ones_like(arrivals2)

	for i in range(1,len(nCount1)):
		nCount1[i] = nCount1[i] + nCount1[i-1]
	for i in range(1,len(nCount2)):
		nCount2[i] = nCount2[i] + nCount2[i-1]

	return nCount1, nCount2
