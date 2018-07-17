import pickle
import numpy as np

cluster_id = "radscan.429463."
job_numbers = 120

def process_file(filename):
	file = open(filename, mode='r')
	lines = [line for line in file]
	# first lines
	epsilon_k0		= float(lines[0].split(" ")[1])
	epsilon_k1		= float(lines[1].split(" ")[1])
	epsilon_k2		= float(lines[2].split(" ")[1])
	epsilon_k3		= float(lines[3].split(" ")[1])
	epsilon_k4		= float(lines[4].split(" ")[1])
	epsilon_k5		= float(lines[5].split(" ")[1])
	epsilon_k6		= float(lines[6].split(" ")[1])
	omega_x0		= float(lines[7].split(" ")[1])
	omega_y0		= float(lines[8].split(" ")[1])
	dx 				= float(lines[9].split(" ")[1])
	epsilon 		= float(lines[10].split(" ")[1])
	max_turns 		= float(lines[11].split(" ")[1])
	from_line 		= float(lines[12].split(" ")[1])
	to_line 		= float(lines[13].split(" ")[1])
	# everything else
	dictionary = {}
	for i in range(14, len(lines)):
		splitted = lines[i].split(" ")
		dictionary[float(splitted[0])] = [int(float(splitted[i])) for i in range(1, len(splitted)-1)]
	#print(dictionary)
	return (epsilon_k0, epsilon_k1, epsilon_k2, epsilon_k3, epsilon_k4, epsilon_k5, epsilon_k6, omega_x0, omega_y0, dx, epsilon, max_turns, from_line, to_line, dictionary)

jobs = []

for i in range(job_numbers):
	print("Processing Job {}.".format(i))
	jobs.append(process_file(cluster_id + str(i) + ".out"))

dictionary = {}

for i in range(job_numbers):
	if (jobs[i][7], jobs[i][8], jobs[i][10]) in dictionary:
		dictionary[(jobs[i][7], jobs[i][8], jobs[i][10])] = {**dictionary[(jobs[i][7], jobs[i][8], jobs[i][10])], **jobs[i][14]}
	else:
		dictionary[(jobs[i][7], jobs[i][8], jobs[i][10])] = jobs[i][14]

for epsilon in dictionary:
	storage = np.zeros((80,80), dtype = int)
	i = -1
	for position in sorted(dictionary[epsilon]):
		i += 1
		storage[i][:len(dictionary[epsilon][position])] = dictionary[epsilon][position]
	storage[storage < 1000] = 0
	for i in range(len(storage)):
		if np.count_nonzero(storage[i]) == 0:
			storage[i:] = 0
	dictionary[epsilon] = storage

#print(dictionary[1])

with open("linscan_dx01_firstonly_dictionary.pkl", "wb") as f:
	pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)