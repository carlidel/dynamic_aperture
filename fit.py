import numpy as np
import matplotlib.pyplot as plt

# Initialize Parameters used in the simulation

dx = 0.01
nangles = 30
angles = np.linspace(0, np.pi/4, nangles)
dtheta = angles[1] - angles[0]
epsilons = [0, 1, 2, 4, 8, 16, 32, 64]
#n_turns = [1000, 2000, 5000, 7000, 10000, 20000, 50000, 70000, 100000, 200000, 500000, 700000, 1000000, 2000000]
n_turns = [1000, 2000, 5000, 7000, 10000, 20000, 50000, 100000, 1000000]

#n_turns = [3000, 4000, 6000, 8000, 30000, 40000, 60000, 80000]

n_turns = np.asarray(n_turns)

# Load data

data = []

for i in range(len(epsilons)):
	temp = []
	for j in range(len(n_turns)):
		scan = np.load("radscan/radscan_T{}_dx{}_nthet{}_epsilon{}.npy".format(n_turns[j],dx,nangles,epsilons[i]))
		temp.append(scan)
	data.append(temp)

# Compute D and Error estimation of D

D = []
Err = []

for i in range(len(epsilons)):
	temp = []
	temp_err = []
	for j in range(len(n_turns)):
		print(data[i][j])
		
		# Basic average for D now...
		temp.append(np.average(data[i][j]))
		
		# Error estimation...
		first_derivatives = np.asarray([(data[i][j][k+1]-data[i][j][k]) / dtheta for k in range(len(data[i][j]) - 1)])
		print(first_derivatives)
		average_derivative = np.average(first_derivatives)
		temp_err.append(np.sqrt(dx * dx / 4 + average_derivative * average_derivative * dtheta * dtheta / 4))

	D.append(temp)
	Err.append(temp_err)

D = np.asarray(D)
Err = np.asarray(Err)

# How's the data?

np.set_printoptions(precision=3)

print(D)
print(Err)

# Non linear Fit

from scipy.optimize import curve_fit

# The k we are going to explore
constant = 1

# The function on which we execute the fit
def func(x, A, B):
	return A + B / np.log10(x)**constant

# The goal is to minimize this
def chi_squared(x, y, sigma, popt, k):
	return (1 / (len(n_turns)-3)) * np.sum(((y - popt[0] - popt[1] / np.log10(x)**k) / sigma)**2)

# For every Epsilon measured...

for i in range(len(D)):
	# Explore different values of k in [-20,20]
	explore_k = []
	for number in np.linspace(-20,20,100):
		constant = number
		popt, pcov = curve_fit(func, n_turns, D[i], sigma = Err[i])
		explore_k.append(chi_squared(n_turns, D[i], Err[i], popt, constant))
	
	# Select Best k and re-execute fit
	explore_k = np.asarray(explore_k)
	print(explore_k.min(), np.linspace(-20,20,100)[explore_k.argmin()])
	
	constant = np.linspace(-20,20,100)[explore_k.argmin()]
	popt, pcov = curve_fit(func, n_turns, D[i], sigma = Err[i])
	
	# Plot everything
	plt.errorbar(n_turns, D[i], yerr = Err[i], linewidth = 0, elinewidth = 2, label = 'Data')
	plt.plot(n_turns, func(n_turns, popt[0], popt[1]), 'g--', label = 'fit: A={:6.3f}, B={:6.3f}, k={:6.3f}'.format(popt[0], popt[1], constant))
	plt.axhline(y = popt[0], color = 'r', linestyle = '-', label = 'y=A={:6.3f}'.format(popt[0]))
	plt.legend()
	plt.xlabel("N turns")
	plt.xscale("log")
	plt.ylabel("D (A.U.)")
	plt.title("Epsilon = {:2.0f}".format(epsilons[i]))
	plt.tight_layout()
	plt.savefig("img/fit_radscan_epsilon{}.png".format(epsilons[i]))
	plt.clf()
