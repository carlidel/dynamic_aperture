import numpy as np
import matplotlib.pyplot as plt

# Load Data

dx = 0.01
nangles = 30
angles = np.linspace(0, np.pi/4, nangles)
epsilons = [0, 1, 2, 4, 8, 16, 32, 64]
n_turns = [1000, 10000, 100000, 1000000]

data = []

for i in range(len(epsilons)):
	temp = []
	for j in range(len(n_turns)):
		scan = np.load("radscan_T{}_dx{}_nthet{}_epsilon{}.npy".format(n_turns[j],dx,nangles,epsilons[i]))
		temp.append(scan)
	data.append(temp)

D = []

for i in range(len(epsilons)):
	temp = []
	for j in range(len(n_turns)):
		#print(data[i][j])
		temp.append(np.average(data[i][j]))
	D.append(temp)

D = np.asarray(D)

np.set_printoptions(precision=2)

print(D)

