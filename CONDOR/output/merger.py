import os

cluster_id = "radscan.411211."
job_numbers = 104

def process_file(filename):
	file = open(filename, mode='r')
	lines = [line for line in file]
	# first 7 lines
	dx = float(lines[0].split(" ")[1])
	n_theta = int(lines[1].split(" ")[1])
	epsilon = float(lines[3].split(" ")[1])
	max_turns = float(lines[4].split(" ")[1])
	from_angle = float(lines[5].split(" ")[1])
	to_angle = float(lines[6].split(" ")[1])
	# everything else
	dictionary = {}
	for i in range(7, len(lines)):
		splitted = lines[i].split(" ")
		dictionary[float(splitted[0])] = [int(float(splitted[i])) for i in range(1, len(splitted)-1)]
	#print(dictionary)
	return (dx, n_theta, epsilon, max_turns, from_angle, to_angle, dictionary)

jobs = []

for i in range(job_numbers):
	print("Processing Job {}.".format(i))
	jobs.append(process_file(cluster_id + str(i) + ".out"))

dictionary = {}

for i in range(job_numbers):
	if jobs[i][2] in dictionary:
		dictionary[jobs[i][2]] = {**dictionary[jobs[i][2]], **jobs[i][6]}
	else:
		dictionary[jobs[i][2]] = jobs[i][6]

print(dictionary[0.0])