import numpy as np
import matplotlib.pyplot as plt
import pickle

# Print precision

np.set_printoptions(precision=3)

# Initialize Parameters used in the simulation

dx = 0.01
nangles = 100
angles = np.linspace(0, np.pi/4, nangles)
dtheta = angles[1] - angles[0]
epsilons = [0, 1, 2, 4, 8, 16, 32, 64]

n_turns = np.array([1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000, 180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])

# Load data

data = pickle.load(open("radscan_dx01_dictionary.pkl", "rb"))

# Compute D and Error estimation of D

D = {}
Err = {}

for epsilon in data:
	temp_D = {}
	temp_Err = {}
	for t in n_turns:
		limit = []
		for line in sorted(data[epsilon]):
			i = 0
			while data[epsilon][line][i] >= t:
				i += 1
			limit.append((i-1) * dx)
		limit = np.asarray(limit)

		# Basic average for D
		temp_D[t] = (np.average(limit))

		# Error estimation
		first_derivatives = np.asarray([(limit[k+1] - limit[k]) / dtheta for k in range(len(limit) - 1)])
		average_derivative = np.average(first_derivatives)
		temp_Err[t] = (np.sqrt(dx * dx / 4 + average_derivative * average_derivative * dtheta * dtheta / 4))

	D[epsilon] = temp_D
	Err[epsilon] = temp_Err 

print(D[1])
print(Err[1])


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

for epsilon in D:
	# Explore different values of k in [-20,20]
	explore_k = []
	for number in np.linspace(-20,20,100):
		constant = number
		popt, pcov = curve_fit(func, [k for k in sorted(D[epsilon])], [D[epsilon][k] for k in sorted(D[epsilon])], sigma = [Err[epsilon][k] for k in sorted(Err[epsilon])])
		explore_k.append(chi_squared([k for k in sorted(D[epsilon])], [D[epsilon][k] for k in sorted(D[epsilon])], [Err[epsilon][k] for k in sorted(Err[epsilon])], popt, constant))
	
	# Select Best k and re-execute fit
	explore_k = np.asarray(explore_k)
	print(explore_k.min(), np.linspace(-20,20,100)[explore_k.argmin()])
	
	constant = np.linspace(-20,20,100)[explore_k.argmin()]
	popt, pcov = curve_fit(func, [k for k in sorted(D[epsilon])], [D[epsilon][k] for k in sorted(D[epsilon])], sigma = [Err[epsilon][k] for k in sorted(Err[epsilon])])
	# Plot everything
	plt.errorbar([k for k in sorted(D[epsilon])], [D[epsilon][k] for k in sorted(D[epsilon])], yerr = [Err[epsilon][k] for k in sorted(Err[epsilon])], linewidth = 0, elinewidth = 2, label = 'Data')
	plt.plot(n_turns, func(n_turns, popt[0], popt[1]), 'g--', label = 'fit: A={:6.3f}, B={:6.3f}, k={:6.3f}'.format(popt[0], popt[1], constant))
	plt.axhline(y = popt[0], color = 'r', linestyle = '-', label = 'y=A={:6.3f}'.format(popt[0]))
	plt.legend()
	plt.xlabel("N turns")
	plt.xscale("log")
	plt.ylabel("D (A.U.)")
	plt.ylim((0,1))
	plt.title("Epsilon = {:2.0f}".format(epsilon))
	plt.tight_layout()
	plt.savefig("img/fit_radscan_epsilon{}.png".format(epsilon))
	plt.clf()
