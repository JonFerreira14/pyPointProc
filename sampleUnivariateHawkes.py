"""
	Here, we first generate a simulated list of arrivals, through tinning

	Then we create a hawke's process for these arrivals and measure the goodness of fit by using the compensator function

	Next, we fit the model using the loglikelihood function, then create the fitted hawke's process and measure the goodness 
	of fit by using the compensator function
"""

## Import the required library
from pyPointProc.hawkesProc import univariateHawkes as uH


if __name__ == "__main__":

	"""
		Simulated hawke's process generation through thinning
	"""
	## Set initial hawke's process variables
	mu = 0.3 	#
	alpha = 0.6	#
	beta = 0.9	#
	
	## Generate a simulated hawkes process list of a arrivals
	simulatedArrivals = uH.thinningFunction(50, mu, alpha, beta, 1)
	print("simulated arrivals:", simulatedArrivals)	

	## Generate a list of hawkes process intensity at each timestamp, 
	intensity, timestamps = uH.hawkesIntensity(mu, alpha, beta, simulatedArrivals, 10, True)

	## Generate compensator values for each arrival
	compensatorValues = uH.compensatorFunction(mu, alpha, beta, simulatedArrivals, True)
	
	## Print the r squared value of the function
	print("r squared is:",uH.goodnessOfFit(compensatorValues, True))

	

	"""
		Loglikelihood optimization of hawke's process
	"""
	## Fit the function
	print('optimizing...')
	mu, alpha, beta = uH.fit(mu, alpha, beta, simulatedArrivals)

	## Create hawke's process with updated mu, alpha, beta
	intensity, timestamps = uH.hawkesIntensity(mu, alpha, beta, simulatedArrivals, 10, True)

	## Check goodness of fit
	compensatorValues = uH.compensatorFunction(mu, alpha, beta, simulatedArrivals, True)
	print("r squared is:",uH.goodnessOfFit(compensatorValues, True))

	