from pyPointProc.hawkesProc import bivariateHawkes as bH

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

	tser1 = [1,2,5,6,12,18,20,21]
	tser2 = [4,5,6,9,12,13,15,18]
	lastArrival = max(tser1[-1], tser2[-1])+1
	hawks1, hawks2, timestamps = bH.hawkesIntensity(tser1, tser2)

	plt.plot(timestamps, hawks1)
	plt.plot(timestamps, hawks2)
	plt.show()

	
