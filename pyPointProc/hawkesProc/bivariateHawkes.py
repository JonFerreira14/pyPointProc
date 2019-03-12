import matplotlib.pyplot as plt 
from math import exp, ceil, log
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import numba

def exponentialKernel(alpha, beta, x, y):
	return (alpha*exp(-beta*(x-y)))

@numba.jit
def hawkesIntensity(arrivals1, arrivals2, params=(0.3,0.1,0.6,0.9,0.2,0.5,1.2,1.0), granularity=10):
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
@numba.jit
def compensatorFunction(arrivals1, arrivals2, params=(0.3,0.1,0.6,0.9,0.2,0.5,1.2,1.0)):
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

@numba.jit(nopython=True)
def logLikelihood(arrivals1, arrivals2, params=(0.3,0.1,0.6,0.9,0.2,0.5,1.2,1.0)):
	## Setting params
	mu1, mu2 = params[0], params[1]
	alpha11, alpha12, alpha21, alpha22 = params[2], params[3], params[4], params[5]
	beta1, beta2 = params[6], params[7]

	T = max(arrivals1[-1], arrivals2[-1])

	## LL = LL(1) + LL(2)
	## LL(1) = -mu1*T + comp val 1+
	## (alpha*exp(-beta*(x-y)))
	## Calc compensator sums
	sum11 = 0
	for i in arrivals1:
		sum11 += (1 - exp(-beta1*(T-i)))
	sum11 = (alpha11/beta1)*sum11

	sum12 = 0
	for i in arrivals2:
		sum12 += (1 - exp(-beta1*(T-i)))
	sum12 = (alpha12/beta1)*sum12

	## Calc recursive sum
	sum13 = 0

	##Calc R11 and R12
	R11, R12 = np.zeros(len(arrivals1)), np.zeros(len(arrivals1))

	for i in range(1, len(R11)):
		R11[int(i)] = exp(-beta1*(arrivals1[int(i)]-arrivals1[int(i-1)]))*(1+R11[int(i-1)])

	for i in range(1, len(R11)):
		tempSum1 = 0

		for j in arrivals2:
			if j >= arrivals1[int(i-1)] and j < arrivals1[int(i)]:
				tempSum1 += exp(-beta1*(arrivals1[int(i)]-j))

		R12[int(i)] = exp(-beta1*(arrivals1[int(i)]-arrivals1[int(i-1)]))*(R12[int(i-1)])+tempSum1

	for i in range(1,len(R11)):
		sum13 += log(mu1 + alpha11*R11[int(i)] + alpha12*R12[int(i)])

	logLikelihood1 = -mu1*T - sum11 - sum12 + sum13


	## LL(2)
	## Calc compensator sums
	sum21 = 0
	for i in arrivals1:
		sum21 += (1 - exp(-beta2*(T-i)))
	sum21 = (alpha21/beta2)*sum21

	sum22 = 0
	for i in arrivals2:
		sum22 += (1 - exp(-beta2*(T-i)))
	sum22 = (alpha22/beta2)*sum22

	## Calc recursive sum
	sum23 = 0

	##Calc R21 and R22
	R21, R22 = np.zeros(len(arrivals2)), np.zeros(len(arrivals2))

	for i in range(1, len(R22)):
		R22[int(i)] = exp(-beta2*(arrivals2[int(i)]-arrivals2[int(i-1)]))*(1+R22[int(i-1)])

	for i in range(1, len(R22)):
		tempSum2 = 0

		for j in arrivals1:
			if j >= arrivals2[int(i-1)] and j < arrivals2[int(i)]:
				tempSum2 += exp(-beta2*(arrivals2[int(i)]-j))

		R21[int(i)] = exp(-beta2*(arrivals1[int(i)]-arrivals1[int(i-1)]))*(R21[int(i-1)])+tempSum2

	for i in range(1,len(R22)):
		sum23 += log(mu2 + alpha21*R21[int(i)] + alpha22*R22[int(i)])


	logLikelihood2 = -mu2*T - sum21 - sum22 + sum23

	logLikelihood = logLikelihood1 + logLikelihood2
	return -logLikelihood

def fit(arrivals1, arrivals2, params=(0.3,0.1,0.6,0.9,0.2,0.5,1.2,1.0)):
	
	def myFitFunc(x):
		tList = []
		for i in x:
			tList.append(min(max(i, 0.00001), 2))

		tList[6] = max(max(tList[2],tList[3])+0.00001,tList[6])
		tList[7] = max(max(tList[4],tList[5])+0.00001,tList[7])

		return logLikelihood(arrivals1, arrivals2, tuple(tList))
	
	x0 = params
	runLimit = 25#50
	iteration = 0
	bestFuncVal = 10000000000
	bestx0 = []

	while iteration < runLimit:
		
		mini = minimize(myFitFunc, x0, method='Nelder-Mead', options={'disp':True})#,'maxiter':10000, 'xtol':10**-20,'ftol':10**-20})
		
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

	tList = []
	for i in bestx0:
		tList.append(min(max(i, 0.00001), 2))
	tList[6] = max(max(tList[2],tList[3])+0.00001,tList[6])
	tList[7] = max(max(tList[4],tList[5])+0.00001,tList[7])

	return tuple(tList)

def cumulativeArrivals(arrivals1, arrivals2):
	nCount1, nCount2 = np.ones_like(arrivals1), np.ones_like(arrivals2)

	for i in range(1,len(nCount1)):
		nCount1[i] = nCount1[i] + nCount1[i-1]
	for i in range(1,len(nCount2)):
		nCount2[i] = nCount2[i] + nCount2[i-1]

	return nCount1, nCount2
