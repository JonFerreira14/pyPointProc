from pyPointProc.hawkesProc import bivariateHawkes as bH
import matplotlib.pyplot as plt


if __name__ == "__main__":
	## Sample arrivals
	tser1 = [1,2,5,6,12,18,20,21]
	tser2 = [4,5,6,9,13,15,18]

	## Hawkes intensity
	hawks1, hawks2, timestamps = bH.hawkesIntensity(tser1, tser2)
	plt.plot(timestamps, hawks1)
	plt.plot(timestamps, hawks2)
	plt.show()

	## Compensator function vs Ncount plots
	arrivalCount1, arrivalCount2 = bH.cumulativeArrivals(tser1, tser2)
	comp1, comp2 = bH.compensatorFunction(tser1, tser2)	
	## Series 1
	plt.plot(tser1, arrivalCount1)
	plt.plot(tser1, comp1)
	plt.show()
	## Series 2
	plt.plot(tser2, arrivalCount2)
	plt.plot(tser2, comp2)
	plt.show()
	
	## Goodness of fit
	print('r squared of series 1 is:', bH.goodnessOfFit(comp1))
	print('r squared of series 2 is:', bH.goodnessOfFit(comp2))

	print("log likelihood is:", bH.logLikelihood(tser1,tser2))

	parameters = bH.fit(tser1,tser2)

	print("new log likelihood is:", bH.logLikelihood(tser1,tser2, parameters))


	comp1, comp2 = bH.compensatorFunction(tser1, tser2, parameters)	
	print('r squared of series 1 is:', bH.goodnessOfFit(comp1))
	print('r squared of series 2 is:', bH.goodnessOfFit(comp2))
	plt.plot(tser1, arrivalCount1)
	plt.plot(tser1, comp1)
	plt.show()
	plt.plot(tser2, arrivalCount2)
	plt.plot(tser2, comp2)
	plt.show()