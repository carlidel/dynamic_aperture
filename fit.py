import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

# Print precision

np.set_printoptions(precision=3)

# Initialize Parameters used in the simulation

dx = 0.01
nangles = 101
angles = np.linspace(0, np.pi/4, nangles)
dtheta = angles[1] - angles[0]
epsilons = [0, 1, 2, 4, 8, 16, 32, 64]

n_turns = np.array([1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000, 180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])

partition_list = np.array([0, np.pi / 8, np.pi / 4])

# Load data

data = pickle.load(open("radscan_dx01_dictionary.pkl", "rb"))

# Compute D and Error estimation of D

def divide_and_compute(data, n_turns, partition_list = [0, np.pi/4]):
    '''
    Given a block of data from a particular epsilon, it computes the D and the
    error for all given turn for the given partition limits.
    A minimum of 2 angles per partition is imposed with an assert.
    '''
    angle = {} # a "dictionary angle" for keeping the angles catalogued
    
    for i in range(len(partition_list) - 1): # for each partition...
        # do the calculations of D and Err
        n_angles = 0
        D = {}
        Err = {}
        for t in n_turns:
            limit = []
            for line in sorted(data):
                if line >= partition_list[i] and line <= partition_list[i+1]:
                    n_angles += 1
                    j = 0
                    while data[line][j] >= t:
                        j += 1
                    limit.append((j - 1) * dx)
            # Minimum angle limit per partition
            assert n_angles >= 2
            
            limit = np.asarray(limit)
            
            # Basic average for D
            D[t] = np.average(limit)
            
            # Error estimation
            first_derivatives = np.asarray([(limit[k+1] - limit[k]) / dtheta for k in range(len(limit) - 1)])
            average_derivative = np.average(first_derivatives)
            Err[t] = (np.sqrt(dx * dx / 4 + average_derivative * average_derivative * dtheta * dtheta / 4))
            
        angle[(partition_list[i] + partition_list[i + 1]) / 2] = (D, Err)
        
    return angle

def function_1(x, A, B, k):
    return A + B / np.log10(x) ** k

def non_linear_fit(data, n_turns, method = 1):
    '''
    Given a tupla in the format (D[], Err[]), it will compute the best fit
    with the given method.
    '''
    if method == 1:
        constant = 1.
        func = lambda x, A, B : A + B / np.log10(x) ** constant
        chi_squared = lambda x, y, sigma, popt, k : (1 / (len(n_turns)-3)) * np.sum(((y - popt[0] - popt[1] / np.log10(x)**k) / sigma)**2)
        # Explore k values in [-20,20]
        explore_k = []
        for number in np.linspace(-20,20,100):
            constant = number
            popt, pcov = curve_fit(func, n_turns, [data[0][i] for i in n_turns], sigma = [data[1][i] for i in n_turns])
            explore_k.append(chi_squared(n_turns, [data[0][i] for i in n_turns], [data[1][i] for i in n_turns], popt, constant))
        
        # Select Best k and re-execute fit
        explore_k = np.asarray(explore_k)
        #print(explore_k.min(), np.linspace(-20,20,100)[explore_k.argmin()])
        constant = np.linspace(-20,20,100)[explore_k.argmin()]
        
        popt, pcov = curve_fit(func, n_turns, [data[0][i] for i in n_turns], sigma = [data[1][i] for i in n_turns])
        return (popt[0], popt[1], constant)
    else:
        print("Method not contemplated.")
        assert False
    
dynamic_aperture = {}

for epsilon in sorted(data):
    dynamic_aperture[epsilon] = divide_and_compute(data[epsilon], n_turns, partition_list)

#print(final_data)

fit_parameters = {}

for epsilon in dynamic_aperture:
    temp = {}
    for angle in dynamic_aperture[epsilon]:
        temp[angle] = non_linear_fit(dynamic_aperture[epsilon][angle], n_turns)
    fit_parameters[epsilon] = temp

# Plot Everything

for epsilon in fit_parameters:
    for angle in fit_parameters[epsilon]:
        plt.errorbar(n_turns, [dynamic_aperture[epsilon][angle][0][i] for i in n_turns], yerr=[dynamic_aperture[epsilon][angle][1][i] for i in n_turns], linewidth = 0, elinewidth = 2, label = 'Data')
        plt.plot(n_turns, function_1(n_turns, fit_parameters[epsilon][angle][0],fit_parameters[epsilon][angle][1],fit_parameters[epsilon][angle][2]), 'g--', label = 'fit: A={:6.3f}, B={:6.3f}, k={:6.3f}'.format(fit_parameters[epsilon][angle][0],fit_parameters[epsilon][angle][1],fit_parameters[epsilon][angle][2]))
        plt.axhline(y = fit_parameters[epsilon][angle][0], color = 'r', linestyle = '-', label = 'y=A={:6.3f}'.format(fit_parameters[epsilon][angle][0]))
        plt.legend()
        plt.xlabel("N turns")
        plt.xscale("log")
        plt.ylabel("D (A.U.)")
        plt.title("dx = {:3.3f}, dtheta = {}, Central angle = {}, Epsilon = {:2.0f}".format(dx, dtheta, angle, epsilon).format(epsilon))
        plt.tight_layout()
        plt.savefig("img/fit_epsilon{}_angle{}_Npart{}.png".format(epsilon,angle,len(partition_list) - 1))
        plt.clf()
        
        

#%%

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
	plt.plot(n_turns, func(n_turns, popt[0], popt[1]), 'g--', label = 'fit: A={:6.3f}, B={:6.3f}, k={:6.3f}'.format(popt[0], popt[1], constant), linewidth = 0.5)
	plt.axhline(y = popt[0], color = 'r', linestyle = '-', label = 'y=A={:6.3f}'.format(popt[0]))
	plt.legend()
	plt.xlabel("N turns")
	#plt.xscale("log")
	plt.ylabel("D (A.U.)")
	plt.ylim((0,1))
	plt.title("dx = {:3.3f}, N_angles = {}, Epsilon = {:2.0f}".format(dx, nangles, epsilon))
	plt.tight_layout()
	plt.savefig("img/fit_radscan_epsilon{}.png".format(epsilon), dpi = 600)
	plt.clf()
