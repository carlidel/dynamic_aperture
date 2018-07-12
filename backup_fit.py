import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
import png_to_jpg as converter

# Print precision and DPI and TEX rendering in plots

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

np.set_printoptions(precision=3)
DPI = 150

# Initialize Parameters used in the simulation

dx = 0.01
nangles = 101
angles = np.linspace(0, np.pi/2, nangles)
dtheta = angles[1] - angles[0]
epsilons = [0, 1, 2, 4, 8, 16, 32, 64]

n_turns = np.array([1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000, 180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])

partition_lists = np.array([[0, np.pi / 2]])

#partition_lists = np.array([[0, np.pi / 2], # Always always keep this one 
 #                           [0, np.pi / 4, np.pi / 2], 
 #                           [0, np.pi / (2 * 3), np.pi / (3), np.pi / 2], 
  #                          [0, np.pi / 8, np.pi * 2 / 8, np.pi * 3 / 8, np.pi / 2],
   #                         [0, np.pi / 10, np.pi * 2 / 10, np.pi * 3 / 10, np.pi * 4 / 10, np.pi / 2],
    #                        [0, np.pi / 12, np.pi * 2 / 12, np.pi * 3 / 12, np.pi * 4 / 12, np.pi * 5 / 12, np.pi / 2]])

# convolve parameters

n_angles = 20
stride = 2

# fit parameters

k_possible_values = np.linspace(-5, 5, 100)
k_error = k_possible_values[1] - k_possible_values[0]

def convolve_and_compute(data, n_turns, n_angles, stride = 1):
    angle = {}
    data = [(sorted(data)[i], data[sorted(data)[i]]) for i in range(len(data))]
    
    for i in range(0, len(data) + 1 - n_angles, stride):
        D = {}
        Err = {}
        for t in n_turns:
            limit = []
            for line in data[i : i + n_angles]:
                j = 0
                while line[1][j] >= t:
                    j += 1
                limit.append((j - 1) * dx)
            limit = np.asarray(limit)
            # Basic average for D
            D[t] = np.average(limit)
            # Error estimation
            first_derivatives = np.asarray([(limit[k+1] - limit[k]) / dtheta for k in range(len(limit) - 1)])
            average_derivative = np.average(first_derivatives)
            Err[t] = (np.sqrt(dx * dx / 4 + average_derivative * average_derivative * dtheta * dtheta / 4))
        angle[(data[i + n_angles - 1][0] + data[i][0]) / 2] = (D, Err)
    return angle

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

def function_2(x, A, B, k):
    return B / (np.log10(x) - A) ** k

def function_3(x, A, B, k):
    return A * (1 + B / (np.log10(x)) ** k)

def non_linear_fit(data, n_turns, method = 1):
    '''
    Given a tupla in the format (D[], Err[]), it will compute the best fit
    with the given method.
    '''
    if method == 1:
        constant = 1.
        func = lambda x, A, B : A + B / np.log10(x) ** constant
        chi_squared = lambda x, y, sigma, popt, k : (1 / (len(n_turns)-3)) * np.sum(((y - popt[0] - popt[1] / np.log10(x)**k) / sigma)**2)
        # Explore k values in [-5,5]
        explore_k = {}
        for number in k_possible_values:
            constant = number
            try:
                popt, pcov = curve_fit(func, n_turns, [data[0][i] for i in n_turns], sigma = [data[1][i] for i in n_turns])
                explore_k[(constant, popt[0], popt[1])]= chi_squared(n_turns, [data[0][i] for i in n_turns], [data[1][i] for i in n_turns], popt, constant)
            except:
                pass
        # Select Best k and fit parameters
        explore_k = sorted(explore_k.items(), key = lambda kv: kv[1])

        # Error of the parameters

        return (explore_k[0][0][1],explore_k[0][0][2],explore_k[0][0][0])
    
    elif method == 2:
        constant = 1.
        func = lambda x, A, B : B / (np.log10(x) - A) ** constant
        chi_squared = lambda x, y, sigma, popt, k : (1 / (len(n_turns)-3)) * np.sum(((y - popt[1] / (np.log10(x) - popt[0])**k) / sigma)**2)
        # Explore k values in [-5,5]
        explore_k = {}
        for number in k_possible_values:
            constant = number
            try:
                popt, pcov = curve_fit(func, n_turns, [data[0][i] for i in n_turns], sigma = [data[1][i] for i in n_turns])
                explore_k[(constant, popt[0], popt[1])]= chi_squared(n_turns, [data[0][i] for i in n_turns], [data[1][i] for i in n_turns], popt, constant)
            except:
                pass
        # Select Best k and fit parameters
        explore_k = sorted(explore_k.items(), key = lambda kv: kv[1])

        # Error of the parameters

        return (explore_k[0][0][1],explore_k[0][0][2],explore_k[0][0][0])
    
    elif method == 3:
        constant = 1.
        func = lambda x, A, B : A * (1 + B / (np.log10(x)) ** constant)
        chi_squared = lambda x, y, sigma, popt, k : (1 / (len(n_turns)-3)) * np.sum(((y - popt[0] * (1 + popt[1] / (np.log10(x))**k)) / sigma)**2)
        # Explore k values in [-5,5]
        explore_k = {}
        for number in k_possible_values:
            constant = number
                
            try:
                popt, pcov = curve_fit(func, n_turns, [data[0][i] for i in n_turns], sigma = [data[1][i] for i in n_turns])
                explore_k[(constant, popt[0], popt[1])]= chi_squared(n_turns, [data[0][i] for i in n_turns], [data[1][i] for i in n_turns], popt, constant)
            except:
                pass
        #print(explore_k.min(), k_possible_values[explore_k.argmin()])
        if len(explore_k) == 0:
            print("Fit failed! Returning zeros...")
            return(0,0,0,0)
        
        # Select Best k and fit parameters
        explore_k = sorted(explore_k.items(), key = lambda kv: kv[1])

        # Error of the parameters

        return (explore_k[0][0][1],explore_k[0][0][2],explore_k[0][0][0])
    
    elif method == 4:
        constant = 1.
        func = lambda x, A, B, C : A + B / ((np.log10(x) - C) ** constant)
        chi_squared = lambda x, y, sigma, popt, k : (1 / (len(n_turns)-3)) * np.sum(((y - popt[0] - (popt[1] / (np.log10(x))**k)) / sigma)**2)
    
    else:
        print("Method not contemplated.")
        assert False
  
#%%
print("Load data")

data = pickle.load(open("radscan_dx01_firstonly_dictionary.pkl", "rb"))


fit_parameters1 = {}    # fit1
fit_parameters2 = {}    # fit2
fit_parameters3 = {}    # fit3
dynamic_aperture = {}   # D with error

for partition_list in partition_lists:
    print(partition_list)
    dyn_temp = {}

    for epsilon in sorted(data):
        dyn_temp[epsilon] = divide_and_compute(data[epsilon], n_turns, partition_list)

    #print(final_data)
    
    # fit1

    fit_parameters = {}

    for epsilon in dyn_temp:
        temp = {}
        for angle in dyn_temp[epsilon]:
            temp[angle] = non_linear_fit(dyn_temp[epsilon][angle], n_turns, method=1)
        fit_parameters[epsilon] = temp
        
    fit_parameters1[len(partition_list)-1] = fit_parameters

    # fit2

    fit_parameters = {}

    for epsilon in dyn_temp:
        temp = {}
        for angle in dyn_temp[epsilon]:
            temp[angle] = non_linear_fit(dyn_temp[epsilon][angle], n_turns, method=2)
        fit_parameters[epsilon] = temp
        
    fit_parameters2[len(partition_list)-1] = fit_parameters
    
    # fit3

    fit_parameters = {}

    for epsilon in dyn_temp:
        temp = {}
        for angle in dyn_temp[epsilon]:
            temp[angle] = non_linear_fit(dyn_temp[epsilon][angle], n_turns, method=3)
        fit_parameters[epsilon] = temp
        
    fit_parameters3[len(partition_list)-1] = fit_parameters

    dynamic_aperture[len(partition_list)-1] = dyn_temp

#%%
print("Plot Everything 1")

N = 1
for epsilon in fit_parameters1[N]:
    for angle in fit_parameters1[N][epsilon]:
        plt.errorbar(n_turns, [dynamic_aperture[N][epsilon][angle][0][i] for i in n_turns], yerr=[dynamic_aperture[N][epsilon][angle][1][i] for i in n_turns], linewidth = 0, elinewidth = 2, label = 'Data')
        plt.plot(n_turns, function_1(n_turns, fit_parameters1[N][epsilon][angle][0],fit_parameters1[N][epsilon][angle][1],fit_parameters1[N][epsilon][angle][2]), 'g--', linewidth=0.5, label = 'fit: $A={:6.3f}, B={:6.3f}, k={:6.3f}$'.format(fit_parameters1[N][epsilon][angle][0],fit_parameters1[N][epsilon][angle][1],fit_parameters1[N][epsilon][angle][2]))
        plt.axhline(y = fit_parameters1[N][epsilon][angle][0], color = 'r', linestyle = '-', label = '$y=A={:6.3f}$'.format(fit_parameters1[N][epsilon][angle][0]))
        plt.xlabel("$N$ turns")
        plt.xscale("log")
        plt.ylabel("$D (A.U.)$")
        plt.ylim(0.,1.)
        plt.title("Fit formula: $A + B / (\log_{10}(x))^k$"+"\n$dx = {:2.2f}, dth = {:3.3f}, central angle = {:3.3f}$,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(dx, dtheta, angle, epsilon[2], epsilon[0], epsilon[1]))
        plt.plot([],[],'', linewidth=0, label = "$A = {:.2} \pm {:.2}$".format(fit_parameters1[N][epsilon][angle][0], np.sqrt(fit_parameters1[N][epsilon][angle][3][0][0])))
        plt.plot([],[],'', linewidth=0, label = "$B = {:.2} \pm {:.2}$".format(fit_parameters1[N][epsilon][angle][1], np.sqrt(fit_parameters1[N][epsilon][angle][3][1][1])))
        plt.plot([],[],'', linewidth=0, label = "$k = {:.2} \pm {:.2}$".format(fit_parameters1[N][epsilon][angle][2], k_error))
        plt.legend()
        plt.tight_layout()
        plt.savefig("img/fit1_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}_angle{:3.3f}_Npart{}.png".format(epsilon[2], epsilon[0], epsilon[1],angle,len(partition_list) - 1), dpi = DPI)
        plt.clf()
            
#%%
print("Plot Everything 2")

N = 1
for epsilon in fit_parameters2[N]:
    for angle in fit_parameters2[N][epsilon]:
        plt.errorbar(n_turns, [dynamic_aperture[N][epsilon][angle][0][i] for i in n_turns], yerr=[dynamic_aperture[N][epsilon][angle][1][i] for i in n_turns], linewidth = 0, elinewidth = 2, label = 'Data')
        plt.plot(n_turns, function_2(n_turns, fit_parameters2[N][epsilon][angle][0],fit_parameters2[N][epsilon][angle][1],fit_parameters2[N][epsilon][angle][2]), 'g--', linewidth=0.5, label = 'fit: $A={:6.3f}, B={:6.3f}, k={:6.3f}$'.format(fit_parameters2[N][epsilon][angle][0],fit_parameters2[N][epsilon][angle][1],fit_parameters2[N][epsilon][angle][2]))
        plt.xlabel("$N$ turns")
        plt.xscale("log")
        plt.ylabel("D (A.U.)")
        plt.ylim(0.,1.)
        plt.title("Fit formula: $B / (log_{10}(x) - A)^k$"+"\n$dx = {:2.2f}, dth = {:3.3f}, c.angle = {:3.3f},$\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(dx, dtheta, angle, epsilon[2], epsilon[0], epsilon[1]))
        plt.plot([],[],'', linewidth=0, label = "$A = {:.2} \pm {:.2}$".format(fit_parameters2[N][epsilon][angle][0], np.sqrt(fit_parameters1[N][epsilon][angle][3][0][0])))
        plt.plot([],[],'', linewidth=0, label = "$B = {:.2} \pm {:.2}$".format(fit_parameters2[N][epsilon][angle][1], np.sqrt(fit_parameters1[N][epsilon][angle][3][1][1])))
        plt.plot([],[],'', linewidth=0, label = "$k = {:.2} \pm {:.2}$".format(fit_parameters2[N][epsilon][angle][2], k_error))
        plt.legend()
        plt.tight_layout()
        plt.savefig("img/fit2_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}_angle{:3.3f}_Npart{}.png".format(epsilon[2], epsilon[0], epsilon[1],angle,len(partition_list) - 1), dpi = DPI)
        plt.clf()

#%%
print("Plot Everything 3")

N = 1
for epsilon in fit_parameters3[N]:
    for angle in fit_parameters3[N][epsilon]:
        plt.errorbar(n_turns, [dynamic_aperture[N][epsilon][angle][0][i] for i in n_turns], yerr=[dynamic_aperture[N][epsilon][angle][1][i] for i in n_turns], linewidth = 0, elinewidth = 2, label = 'Data')
        plt.plot(n_turns, function_3(n_turns, fit_parameters3[N][epsilon][angle][0],fit_parameters3[N][epsilon][angle][1],fit_parameters3[N][epsilon][angle][2]), 'g--', linewidth=0.5, label = 'fit: $A={:6.3f}, B={:6.3f}, k={:6.3f}$'.format(fit_parameters3[N][epsilon][angle][0],fit_parameters3[N][epsilon][angle][1],fit_parameters3[N][epsilon][angle][2]))
        plt.axhline(y = fit_parameters3[N][epsilon][angle][0], color = 'r', linestyle = '-', label = 'y=A={:6.3f}'.format(fit_parameters3[N][epsilon][angle][0]))
        plt.xlabel("$N$ turns")
        plt.xscale("log")
        plt.ylabel("D (A.U.)")
        plt.ylim(0.,1.)
        plt.title("Fit formula: $A * (1 + B / (log{10}(x) - C)^k)$"+"\n$dx = {:2.2f}, dth = {:3.3f}, c.angle = {:3.3f},$\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(dx, dtheta, angle, epsilon[2], epsilon[0], epsilon[1]))
        plt.plot([],[],'', linewidth = 0, label = "$A = {:.2} \pm {:.2}$".format(fit_parameters3[N][epsilon][angle][0], np.sqrt(fit_parameters1[N][epsilon][angle][3][0][0])))
        plt.plot([],[],'', linewidth = 0, label = "$B = {:.3} \pm {:.3}$".format(fit_parameters3[N][epsilon][angle][1], np.sqrt(fit_parameters1[N][epsilon][angle][3][1][1])))
        plt.plot([],[],'', linewidth = 0, label = "$k = {:.2} \pm {:.2}$".format(fit_parameters3[N][epsilon][angle][2], k_error))
        plt.legend()
        plt.tight_layout()
        plt.savefig("img/fit3_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}_angle{:3.3f}_Npart{}.png".format(epsilon[2], epsilon[0], epsilon[1],angle,len(partition_list) - 1), dpi = DPI)
        plt.clf()

#%%
print("Fit Parameter Comparison 1")

for sector in fit_parameters1:
    if sector > 1:
        for epsilon in fit_parameters1[sector]:    
            theta = []
            A = []
            B = []
            k = []
            for angle in fit_parameters1[sector][epsilon]:
                theta.append(angle / np.pi)
                A.append(fit_parameters1[sector][epsilon][angle][0])
                B.append(fit_parameters1[sector][epsilon][angle][1])
                k.append(fit_parameters1[sector][epsilon][angle][2])
            plt.plot(theta, A, marker = "o", linewidth = 0.5, label = "A")
            plt.plot(theta, B, marker = "*", linewidth = 0.5, label = "B")
            plt.plot(theta, k, marker = "^", linewidth = 0.5, label = "k")
            plt.xlim((0,0.5))
            plt.xlabel("Theta $(rad / \pi)$")
            plt.ylabel("Fit values (A.U.)")
            plt.title("Fit 1 values at different angles $(nsectors = {})$,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(sector, epsilon[2], epsilon[0], epsilon[1]))
            plt.legend()
            plt.tight_layout()
            plt.savefig("img/angles1_nsec{}_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(sector, epsilon[2], epsilon[0], epsilon[1]), dpi = DPI)
            plt.clf()

#%%
print("Fit Parameter Comparison 2")

for sector in fit_parameters2:
    if sector > 1:
        for epsilon in fit_parameters2[sector]:
            theta = []
            A = []
            B = []
            k = []    
            for angle in fit_parameters2[sector][epsilon]:
                theta.append(angle / np.pi)
                A.append(fit_parameters2[sector][epsilon][angle][0])
                B.append(fit_parameters2[sector][epsilon][angle][1])
                k.append(fit_parameters2[sector][epsilon][angle][2])
            plt.plot(theta, A, marker = "o", linewidth = 0.5, label = "A")
            plt.plot(theta, B, marker = "*", linewidth = 0.5, label = "B")
            plt.plot(theta, k, marker = "^", linewidth = 0.5, label = "k")
            plt.xlim((0,0.5))
            plt.xlabel("Theta $(rad / \pi)$")
            plt.ylabel("Fit values (A.U.)")
            plt.title("Fit 2 values at different angles $(nsectors = {})$,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(sector, epsilon[2], epsilon[0], epsilon[1]))
            plt.legend()
            plt.tight_layout()
            plt.savefig("img/angles2_nsec{}_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(sector, epsilon[2], epsilon[0], epsilon[1]), dpi = DPI)
            plt.clf()

#%%
print("Fit Parameter Comparison 3")

for sector in fit_parameters3:
    if sector > 1:
        for epsilon in fit_parameters3:    
            theta = []
            A = []
            B = []
            k = []
            for angle in fit_parameters3[sector][epsilon]:
                theta.append(angle / np.pi)
                A.append(fit_parameters3[sector][epsilon][angle][0])
                B.append(fit_parameters3[sector][epsilon][angle][1])
                k.append(fit_parameters3[sector][epsilon][angle][2])
            plt.plot(theta, A, marker = "o", linewidth = 0.5, label = "A")
            plt.plot(theta, B, marker = "*", linewidth = 0.5, label = "B")
            plt.plot(theta, k, marker = "^", linewidth = 0.5, label = "k")
            plt.xlim((0,0.5))
            plt.xlabel("Theta $(rad / \pi)$")
            plt.ylabel("Fit values (A.U.)")
            plt.title("Fit 3 values at different angles $(nsectors = {})$,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(sector, epsilon[2], epsilon[0], epsilon[1]))
            plt.legend()
            plt.tight_layout()
            plt.savefig("img/angles3_nsec{}_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(sector, epsilon[2], epsilon[0], epsilon[1]), dpi = DPI)
            plt.clf()

#%%
print("Big Guns")

conv_dynamic_aperture = {}
for epsilon in data:
    conv_dynamic_aperture[epsilon] = convolve_and_compute(data[epsilon], n_turns, n_angles, stride)

conv_fit_parameters1 = {}    # fit1
conv_fit_parameters2 = {}    # fit2
conv_fit_parameters3 = {}    # fit3

print("fit 1")
for epsilon in conv_dynamic_aperture:
    temp = {}
    for angle in conv_dynamic_aperture[epsilon]:
        temp[angle] = non_linear_fit(conv_dynamic_aperture[epsilon][angle], n_turns, method = 1)
    conv_fit_parameters1[epsilon] = temp

print("fit 2")
for epsilon in conv_dynamic_aperture:
    temp = {}
    for angle in conv_dynamic_aperture[epsilon]:
        temp[angle] = non_linear_fit(conv_dynamic_aperture[epsilon][angle], n_turns, method = 2)
    conv_fit_parameters2[epsilon] = temp

print("fit 3")
for epsilon in conv_dynamic_aperture:
    temp = {}
    for angle in conv_dynamic_aperture[epsilon]:
        temp[angle] = non_linear_fit(conv_dynamic_aperture[epsilon][angle], n_turns, method = 3)
    conv_fit_parameters3[epsilon] = temp

#%%
print("Fit Parameter Comparison 1_bis")

for epsilon in conv_fit_parameters1:    
    theta = []
    A = []
    B = []
    k = []
    for angle in conv_fit_parameters1[epsilon]:
        theta.append(angle / np.pi)
        A.append(conv_fit_parameters1[epsilon][angle][0])
        B.append(conv_fit_parameters1[epsilon][angle][1])
        k.append(conv_fit_parameters1[epsilon][angle][2])
    plt.plot(theta, A, marker = "o", linewidth = 0.5, label = "A")
    plt.plot(theta, B, marker = "*", linewidth = 0.5, label = "B")
    plt.plot(theta, k, marker = "^", linewidth = 0.5, label = "k")
    plt.xlim((0,0.5))
    plt.xlabel("Theta $(rad / \pi)$")
    plt.ylabel("Fit values (A.U.)")
    plt.title("Fit 1 values at different angles,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(epsilon[2], epsilon[0], epsilon[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/many_angles1_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(epsilon[2], epsilon[0], epsilon[1]), dpi = DPI)
    plt.clf()

#%%
print("Fit Parameter Comparison 2_bis")

for epsilon in conv_fit_parameters2:
    theta = []
    A = []
    B = []
    k = []    
    for angle in conv_fit_parameters2[epsilon]:
        theta.append(angle / np.pi)
        A.append(conv_fit_parameters2[epsilon][angle][0])
        B.append(conv_fit_parameters2[epsilon][angle][1])
        k.append(conv_fit_parameters2[epsilon][angle][2])
    plt.plot(theta, A, marker = "o", linewidth = 0.5, label = "A")
    plt.plot(theta, B, marker = "*", linewidth = 0.5, label = "B")
    plt.plot(theta, k, marker = "^", linewidth = 0.5, label = "k")
    plt.xlim((0,0.5))
    plt.xlabel("Theta $(rad / \pi)$")
    plt.ylabel("Fit values (A.U.)")
    plt.title("Fit 2 values at different angles,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(epsilon[2], epsilon[0], epsilon[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/many_angles2_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(epsilon[2], epsilon[0], epsilon[1]), dpi = DPI)
    plt.clf()

#%%
print("Fit Parameter Comparison 3_bis")

for epsilon in conv_fit_parameters3:    
    theta = []
    A = []
    B = []
    k = []
    for angle in conv_fit_parameters3[epsilon]:
        theta.append(angle / np.pi)
        A.append(conv_fit_parameters3[epsilon][angle][0])
        B.append(conv_fit_parameters3[epsilon][angle][1])
        k.append(conv_fit_parameters3[epsilon][angle][2])
    plt.plot(theta, A, marker = "o", linewidth = 0.5, label = "A")
    plt.plot(theta, B, marker = "*", linewidth = 0.5, label = "B")
    plt.plot(theta, k, marker = "^", linewidth = 0.5, label = "k")
    plt.xlim((0,0.5))
    plt.xlabel("Theta $(rad / \pi)$")
    plt.ylabel("Fit values (A.U.)")
    plt.title("Fit 3 values at different angles,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(sector, epsilon[2], epsilon[0], epsilon[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/many_angles3_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(epsilon[2], epsilon[0], epsilon[1]), dpi = DPI)
    plt.clf()


#%%
print("Draw 2D stability Maps")

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
        plt.fill(x, y, label="$turns = {}$".format(level))
    plt.legend()
    plt.xlabel("X coordinate (A.U.)")
    plt.ylabel("Y coordinate (A.U.)")
    plt.xlim(0,0.8)
    plt.ylim(0,0.7)
    plt.title("Stable Region\n$(\omega_x = {:3.3f}, \omega_y = {:3.3f}, \epsilon = {:3.3f})$".format(key[0], key[1], key[2]))
    plt.tight_layout()
    plt.savefig("img/stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(key[2], key[0], key[1]), dpi = DPI)
    plt.clf()
    
#%%
print("Convert to JPEG")

converter.png_to_jpg("img/")
