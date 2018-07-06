import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

# Print precision

np.set_printoptions(precision=3)

# Initialize Parameters used in the simulation

dx = 0.01
nangles = 101
angles = np.linspace(0, np.pi/2, nangles)
dtheta = angles[1] - angles[0]
epsilons = [0, 1, 2, 4, 8, 16, 32, 64]

n_turns = np.array([1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000, 180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])

partition_lists = np.array([[0, np.pi / 2], [0, np.pi / 4, np.pi / 2], [0, np.pi / (2 * 3), np.pi / (3), np.pi / 2], [0, np.pi / 8, np.pi * 2 / 8, np.pi * 3 / 8, np.pi / 2]])

# Load data

data = pickle.load(open("radscan_dx01_wxy_dictionary.pkl", "rb"))

# Compute D and Error estimation of D

def divide_and_compute(data, n_turns, partition_list = [0, np.pi/2]):
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

# Actual execution

partition = {}

for partition_list in partition_lists:
    print(partition_list)
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
        
    partition[len(partition_list)-1] = fit_parameters

#%%
# Plot Everything

for N in partition:
    for epsilon in partition[N]:
        for angle in partition[N][epsilon]:
            plt.errorbar(n_turns, [dynamic_aperture[epsilon][angle][0][i] for i in n_turns], yerr=[dynamic_aperture[epsilon][angle][1][i] for i in n_turns], linewidth = 0, elinewidth = 2, label = 'Data')
            plt.plot(n_turns, function_1(n_turns, partition[N][epsilon][angle][0],partition[N][epsilon][angle][1],partition[N][epsilon][angle][2]), 'g--', label = 'fit: A={:6.3f}, B={:6.3f}, k={:6.3f}'.format(partition[N][epsilon][angle][0],partition[N][epsilon][angle][1],partition[N][epsilon][angle][2]))
            plt.axhline(y = partition[N][epsilon][angle][0], color = 'r', linestyle = '-', label = 'y=A={:6.3f}'.format(partition[N][epsilon][angle][0]))
            plt.legend()
            plt.xlabel("N turns")
            plt.xscale("log")
            plt.ylabel("D (A.U.)")
            plt.ylim(0.,1.)
            plt.title("dx = {:3.3f}, dth = {:3.3f}, c.angle = {:3.3f},\nepsilon = {:2.0f}, wx = {:2.2f}, wy = {:2.2f}".format(dx, dtheta, angle, epsilon[2], epsilon[0], epsilon[1]))
            plt.tight_layout()
            plt.savefig("img/fit_eps{:2.0f}_wx{:2.2f}_wy{:2.2f}_angle{:3.3f}_Npart{}.png".format(epsilon[2], epsilon[0], epsilon[1],angle,len(partition_list) - 1), dpi = 600)
            plt.clf()
            
#%%
# Fit Parameter Comparison

for epsilon in partition[1]:
    theta = []
    A = []
    B = []
    k = []    
    for sector in partition:
        for angle in partition[sector][epsilon]:
            theta.append(angle)
            A.append(partition[sector][epsilon][angle][0])
            B.append(partition[sector][epsilon][angle][1])
            k.append(partition[sector][epsilon][angle][2])
    plt.plot(theta, A, "o", label = "A")
    plt.plot(theta, B, "*", label = "B")
    plt.plot(theta, k, "^", label = "k")
    plt.xlabel("Theta (radians)")
    plt.ylabel("Fit values (A.U.)")
    plt.title("Fit values at different angles,\nepsilon = {:2.0f}, wx = {:2.2f}, wy = {:2.2f}".format(epsilon[2], epsilon[0], epsilon[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/angles_eps{:2.0f}_wx{:2.2f}_wy{:2.2f}.png".format(epsilon[2], epsilon[0], epsilon[1]), dpi = 600)
    plt.clf()

#%%
# Draw 2D stability Maps
stability_levels = np.array([1000, 10000, 100000, 1000000, 10000000])

for key in data:
    for level in stability_levels:
        x = []
        y = []
        x.append(0.)
        y.append(0.)
        for line in sorted(data[key]):
            j = 0
            while data[key][line][j] >= level:
                j += 1
            x.append((j - 1) * dx * np.cos(line))
            y.append((j - 1) * dx * np.sin(line))
        plt.fill(x, y, label="N_turns = {}".format(level))
    plt.legend()
    plt.tight_layout()
    plt.show()









