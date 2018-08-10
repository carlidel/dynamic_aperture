import numpy as np
import pickle

def process_file(filename):
	file = open(filename, mode='r')
	lines = [line for line in file]
	times = []
	intensities = []
	for i in range(len(lines)):
		times.append(float(lines[i][3:13]))
		intensities.append(float(lines[i][19:29]))
	times = np.asarray(times)
	intensities = np.asanyarray(intensities)
	return (times, intensities)

dictionary = {}
dictionary["6a"] = process_file("fig6a.dat")
dictionary["6b"] = process_file("fig6b.dat")
dictionary["7a"] = process_file("fig7a.dat")
dictionary["7b"] = process_file("fig7b.dat")
dictionary["7c"] = process_file("fig7c.dat")
dictionary["7d"] = process_file("fig7d.dat")

with open("data_nek_dictionary.pkl", "wb") as f:
	pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)