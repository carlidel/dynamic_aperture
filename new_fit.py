import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle
from scipy.optimize import curve_fit
import png_to_jpg as converter

# Print precision and DPI precision and TEX rendering in plots

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

np.set_printoptions(precision=3)
DPI = 300

# Parameters placed in the simulation

dx = 0.01
n_scanned_angles = 101
angles = np.linspace(0, np.pi / 2, n_scanned_angles + 1)
dtheta = angles[1] - angles[0]

# Scanned N_turns

'''n_turns = np.array([1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000, 180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])'''

n_turns = np.array([1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 12000, 14000, 16000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 120000, 140000, 160000, 180000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000, 950000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000, 6500000, 7000000, 7500000, 8000000, 8500000, 9000000, 9500000, 10000000])

# Partition list for basic angle partitioning

partition_lists = [[0, np.pi / 2]]

partition_lists = [[0, np.pi / 2], # Always always keep this one 
                   [0, np.pi / 4, np.pi / 2], 
                   [0, np.pi / (2 * 3), np.pi / (3), np.pi / 2], 
                   [0, np.pi / 8, np.pi * 2 / 8, np.pi * 3 / 8, np.pi / 2],
                   [0, np.pi / 10, np.pi * 2 / 10, np.pi * 3 / 10, np.pi * 4 / 10, np.pi / 2],
                   [0, np.pi / 12, np.pi * 2 / 12, np.pi * 3 / 12, np.pi * 4 / 12, np.pi * 5 / 12, np.pi / 2]
                  ]


# Convolution parameters for advanced angle paritioning

n_angles_in_partition = 30
stride = 5

# Exponential fit parameters

k_max = 5.
k_min = -5.
n_k = 200
k_possible_values = np.linspace(k_min, k_max, n_k)
k_error = k_possible_values[1] - k_possible_values[0]

# Function Definitions

def compute_D(limit):
    return np.average(limit)

def error_estimation(limit):
    first_derivatives = np.asarray([(limit[k+1] - limit[k]) / dtheta for k in range(len(limit) - 1)])
    average_derivative = np.average(first_derivatives)
    return (np.sqrt(dx * dx / 4 + average_derivative * average_derivative * dtheta * dtheta / 4))

def convolve_and_compute(data, n_turns, n_angles_in_partition, stride = 1):
    angle = {}
    data = [(sorted(data)[i], data[sorted(data)[i]]) for i in range(len(data))]
    for i in range(0, len(data) + 1 - n_angles_in_partition, stride):
        D = {}
        Err = {}
        for t in n_turns:
            limit = []
            for line in data[i : i + n_angles_in_partition]:
                j = 0
                while line[1][j] >= t:
                    j += 1
                limit.append((j - 1) * dx)
            limit = np.asarray(limit)
            D[t] = compute_D(limit)
            Err[t] = error_estimation(limit)
        angle[(data[i + n_angles_in_partition - 1][0] + data[i][0]) / 2] = (D, Err)
    return angle

def divide_and_compute(data, n_turns, partition_list = [0, np.pi/2]):
    angle = {}
    for i in range(len(partition_list) - 1): 
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
            assert n_angles >= 5
            limit = np.asarray(limit)
            D[t] = compute_D(limit)
            Err[t] = error_estimation(limit)           
        angle[(partition_list[i] + partition_list[i + 1]) / 2] = (D, Err)
    return angle

function_1k = lambda x, A, B, k : A + B / np.log10(x) ** k

function_2k = lambda x, A, B, k : B / (np.log10(x) - A) ** k

function_3k = lambda x, A, B, k : A * (1 + B / (np.log10(x)) ** k)

function_4k = lambda x, A, B, C, k : A + B / (np.log10(x) - C) ** k

def non_linear_fit(data, n_turns, method = 1):
    function_1 = lambda x, A, B : A + B / np.log10(x) ** k
    chi_squared_1 = lambda x, y, sigma, popt : (1 / (len(n_turns) - 3)) * np.sum(((y - function_1(x, popt[0], popt[1])) / sigma)**2)
    
    function_2 = lambda x, A, B : B / (np.log10(x) - A) ** k   
    chi_squared_2 = lambda x, y, sigma, popt : (1 / (len(n_turns) - 3)) * np.sum(((y - function_2(x, popt[0], popt[1])) / sigma)**2)
    
    function_3 = lambda x, A, B : A * (1 + B / (np.log10(x)) ** k)
    chi_squared_3 = lambda x, y, sigma, popt : (1 / (len(n_turns) - 3)) * np.sum(((y - function_3(x, popt[0], popt[1])) / sigma)**2)
    
    function_4 = lambda x, A, B, C : A + B / (np.log10(x) - C) ** k   
    chi_squared_4 = lambda x, y, sigma, popt : (1 / (len(n_turns) - 3)) * np.sum(((y - function_4(x, popt[0], popt[1], popt[2])) / sigma)**2)

    if method == 1:
        explore_k = {}
        for number in k_possible_values:
            k = number
            try:
                popt, pcov = curve_fit(function_1, n_turns, [data[0][i] for i in n_turns], p0=[0.,1.], sigma=[data[1][i] for i in n_turns])
                explore_k[k] = (popt, pcov, (chi_squared_1(n_turns, [data[0][i] for i in n_turns], [data[1][i] for i in n_turns], popt)))
            except RuntimeError:
                #print(k, "error1")
                pass
        #print(len(explore_k))
        assert len(explore_k) > 0
        return explore_k
    elif method == 2:
        explore_k = {}
        # FACT: IT DOES NOT CONVERGE WITH k < 0!!! JUST WASTED TIME..
        for number in k_possible_values:
            k = number
            try:
                popt, pcov = curve_fit(function_2, n_turns, [data[0][i] for i in n_turns], p0=[0.,1.], sigma=[data[1][i] for i in n_turns])
                explore_k[k] = (popt, pcov, chi_squared_2(n_turns, [data[0][i] for i in n_turns], [data[1][i] for i in n_turns], popt))
            except RuntimeError:
                #print(k, "error2")
                pass
        #print(len(explore_k))
        assert len(explore_k) > 0
        return explore_k
    elif method == 3:
        explore_k = {}
        for number in k_possible_values:
            k = number
            try:
                popt, pcov = curve_fit(function_3, n_turns, [data[0][i] for i in n_turns], p0=[0.,1.], sigma=[data[1][i] for i in n_turns])
                explore_k[k] = (popt, pcov, chi_squared_3(n_turns, [data[0][i] for i in n_turns], [data[1][i] for i in n_turns], popt))
            except RuntimeError:
                #print(k, "error3")
                pass
        #print(len(explore_k))
        assert len(explore_k) > 0
        return explore_k
    elif method == 4:
        explore_k = {}
        for number in k_possible_values:
            k = number
            try:
                popt, pcov = curve_fit(function_4, n_turns, [data[0][i] for i in n_turns], p0=[0.,1.,1.], sigma=[data[1][i] for i in n_turns])
                explore_k[k] = (popt, pcov, chi_squared_4(n_turns, [data[0][i] for i in n_turns], [data[1][i] for i in n_turns], popt))
            except RuntimeError:
                #print(k, "error4")
                pass
        #print(len(explore_k))
        assert len(explore_k) > 0
        return explore_k
    else:
        print("No method")
        assert False

def select_best_fit(parameters, params_are_4 = False):
	best = sorted(parameters.items(), key = lambda kv: kv[1][2])[0]
	if not params_are_4:
		return(best[1][0][0], np.sqrt(best[1][1][0][0]), best[1][0][1], np.sqrt(best[1][1][1][1]), best[0])
	else:
		return(best[1][0][0], np.sqrt(best[1][1][0][0]), best[1][0][1], np.sqrt(best[1][1][1][1]), best[1][0][2], np.sqrt(best[1][1][2][2]), best[0])

#%%
print("Load data")

data = pickle.load(open("radscan_dx01_firstonly_dictionary.pkl", "rb"))
lin_data = pickle.load(open("linscan_dx01_firstonly_dictionary.pkl", "rb"))

#%%
print("Fit on basic partitions")

fit_parameters1 = {}    # fit1
fit_parameters2 = {}    # fit2
fit_parameters3 = {}    # fit3
fit_parameters4 = {}    # fit4
dynamic_aperture = {}   # D with error

for partition_list in partition_lists:
    print(partition_list)
    dyn_temp = {}

    for epsilon in sorted(data):
        dyn_temp[epsilon] = divide_and_compute(data[epsilon], n_turns, partition_list)
        
    dynamic_aperture[len(partition_list)-1] = dyn_temp

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
    
    # fit4

    fit_parameters = {}

    for epsilon in dyn_temp:
        temp = {}
        for angle in dyn_temp[epsilon]:
            temp[angle] = non_linear_fit(dyn_temp[epsilon][angle], n_turns, method=4)
        fit_parameters[epsilon] = temp
        
    fit_parameters4[len(partition_list)-1] = fit_parameters

#%%
def plot_fit_basic(numfit, best_fit, N, epsilon, angle, n_turns, dynamic_aperture, func, y_bar = False, params_are_4 = False):
    plt.errorbar(n_turns, [dynamic_aperture[N][epsilon][angle][0][i] for i in n_turns], yerr=[dynamic_aperture[N][epsilon][angle][1][i] for i in n_turns], linewidth = 0, elinewidth = 2, label = 'Data')
    if not params_are_4:
        plt.plot(n_turns, func(n_turns, best_fit[0], best_fit[2], best_fit[4]), 'g--', linewidth=0.5, label = 'fit: $A={:6.3f}, B={:6.3f}, k={:6.3f}$'.format(best_fit[0], best_fit[2], best_fit[4]))
    else:
        plt.plot(n_turns, func(n_turns, best_fit[0], best_fit[2], best_fit[4], best_fit[6]), 'g--', linewidth=0.5, label = 'fit: $A={:6.3f}, B={:6.3f}, C={:6.3f}, k={:6.3f}$'.format(best_fit[0], best_fit[2], best_fit[4], best_fit[6]))
    if y_bar:
    	plt.axhline(y = best_fit[0], color = 'r', linestyle = '-', label = '$y=A={:6.3f}$'.format(best_fit[0]))
    plt.xlabel("$N$ turns")
    plt.xscale("log")
    plt.ylabel("$D (A.U.)$")
    plt.ylim(0.,1.)
    plt.title("Fit formula {},\n$dx = {:2.2f}, dth = {:3.2f}, mid\,angle = {:3.3f}$,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(numfit, dx, dtheta, angle, epsilon[2], epsilon[0], epsilon[1]))
    plt.plot([],[],'', linewidth=0, label = "$A = {:.2} \pm {:.2}$".format(best_fit[0], best_fit[1]))
    plt.plot([],[],'', linewidth=0, label = "$B = {:.2} \pm {:.2}$".format(best_fit[2], best_fit[3]))
    if not params_are_4:
    	plt.plot([],[],'', linewidth=0, label = "$k = {:.2} \pm {:.2}$".format(best_fit[4], k_error))
    else:
    	plt.plot([],[],'', linewidth=0, label = "$C = {:.2} \pm {:.2}$".format(best_fit[4], best_fit[5]))
    	plt.plot([],[],'', linewidth=0, label = "$k = {:.2} \pm {:.2}$".format(best_fit[6], k_error))
    plt.legend(prop={"size" : 7})
    plt.tight_layout()
    plt.savefig("img/fit{}_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}_angle{:3.3f}_Npart{}.png".format(numfit, epsilon[2], epsilon[0], epsilon[1],angle,1), dpi = DPI) ## for now partition list is set to 1
    plt.clf()


print("Plot fit 1, only partition 1")
N = 1

for epsilon in fit_parameters1[N]:
	for angle in fit_parameters1[N][epsilon]:
		best_fit = select_best_fit(fit_parameters1[N][epsilon][angle])
		plot_fit_basic(1, best_fit, N, epsilon, angle, n_turns, dynamic_aperture, function_1k, True)

print("Plot fit 2, only partition 1")
N = 1

for epsilon in fit_parameters2[N]:
	for angle in fit_parameters2[N][epsilon]:
		best_fit = select_best_fit(fit_parameters2[N][epsilon][angle])
		plot_fit_basic(2, best_fit, N, epsilon, angle, n_turns, dynamic_aperture, function_2k, False)

print("Plot fit 3, only partition 1")
N = 1

for epsilon in fit_parameters3[N]:
	for angle in fit_parameters3[N][epsilon]:
		best_fit = select_best_fit(fit_parameters3[N][epsilon][angle])
		plot_fit_basic(3, best_fit, N, epsilon, angle, n_turns, dynamic_aperture, function_3k, True)

print("Plot fit 4, only partition 1, 4 parameters")
N = 1

for epsilon in fit_parameters4[N]:
    for angle in fit_parameters4[N][epsilon]:
        best_fit = select_best_fit(fit_parameters4[N][epsilon][angle], True)
        plot_fit_basic(4, best_fit, N, epsilon, angle, n_turns, dynamic_aperture, function_4k, True, True)
        
#%%
print("Plot Fit Performances.")        
        
def compare_fit_chi_squared(fit1, fit2, fit3, fit4):
    for epsilon in fit1[1]:
        for angle in fit1[1][epsilon]:
            plt.plot(list(fit1[1][epsilon][angle].keys()), [x[2] for x in list(fit1[1][epsilon][angle].values())], marker = "o", markersize = 0.5,  linewidth = 0.5, label = "fit1")
            plt.plot(list(fit2[1][epsilon][angle].keys()), [x[2] for x in list(fit2[1][epsilon][angle].values())], marker = "o",markersize = 0.5, linewidth= 0.5, label = "fit2")
            #plt.plot(list(fit3[1][epsilon][angle].keys()), [x[2] for x in list(fit3[1][epsilon][angle].values())], marker = "o",markersize = 0.5, linewidth = 0.5, label = "fit3")
            plt.plot(list(fit4[1][epsilon][angle].keys()), [x[2] for x in list(fit4[1][epsilon][angle].values())], marker = "o",markersize = 0.5, linewidth = 0.5, label = "fit4")
            plt.xlabel("k value")
            plt.ylabel("Chi-Squared value")
            plt.title("Fit Performance Comparison (Chi-Squared based), $\epsilon = {}$".format(epsilon[2]))
            plt.legend()
            plt.tight_layout()
            plt.savefig("img/fit_performance_comparison_epsilon{}.png".format(epsilon[2]), dpi = DPI)
            plt.clf()

compare_fit_chi_squared(fit_parameters1, fit_parameters2, fit_parameters3, fit_parameters4)

#%%
print("Is This Loss?")

from scipy import integrate

# Intensity functions

def intensity_zero(x, y, sigma_x = 1, sigma_y = 1):
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-((x**2/(2*sigma_x**2))+(y**2/(2*sigma_y**2))))

def relative_intensity_D_law(D):
    return 1 - np.exp(- D**2 / 2)

def grid_intensity(grid):
    # Integrare con Simpson prima in un verso, poi nell'altro
    return integrate.trapz(np.asarray([integrate.simps(line, dx=dx) for line in grid]), dx=dx)
    # Oppure sommare senza pietà?
    #return np.sum(grid)
    

# Weights at beginning
weights = np.array([[intensity_zero(x * dx, y * dx) for x in range(80)] for y in range(80)])

# PRECISE

loss_precise = {}
for epsilon in lin_data:
    intensity_evolution = [0.25]
    for time in n_turns:
        mask = np.copy(lin_data[epsilon])
        mask[mask < time] = 0
        mask[mask >= time] = 1
        masked_weights = weights * mask
        intensity_evolution.append(grid_intensity(masked_weights))
    loss_precise[epsilon] = np.asarray(intensity_evolution) * 4

# D generalized
loss_D = []

def select_angle(x, y, partition):
    for i in range(len(partition) - 1):
        if np.arctan2(y, x) <= partition[i + 1]:
            return (partition[i] + partition[i+1]) / 2

for partition in partition_lists:
    temp_loss = {}
    for epsilon in dynamic_aperture[len(partition) - 1]:
        intensity_evolution = [0.25]
        for time in n_turns:
            mask = np.array([[(x*x*dx*dx + y*y*dx*dx) <= dynamic_aperture[len(partition) - 1][epsilon][select_angle(x,y,partition)][0][time] for x in range(80)] for y in range(80)], dtype = int)
            masked_weights = weights * mask
            intensity_evolution.append(grid_intensity(masked_weights))
        temp_loss[epsilon] = np.asarray(intensity_evolution) * 4
    loss_D.append(temp_loss)

#%%
print("Plot Loss.")

for epsilon in loss_D[0]:
    plt.plot(np.concatenate((np.array([0]),n_turns)), loss_precise[epsilon], linewidth = 0.5, label="Precise loss")
    plt.plot(np.concatenate((np.array([0]),n_turns)), loss_D[0][epsilon], linewidth = 0.5, label="D computed loss")
    plt.plot(np.concatenate((np.array([0]),n_turns)), loss_D[1][epsilon], linewidth = 0.5, label="D computed loss (parted in 2 angles)")
    plt.plot(np.concatenate((np.array([0]),n_turns)), loss_D[2][epsilon], linewidth = 0.5, label="D computed loss (parted in 3 angles)")
    plt.xlabel("N turns")
    plt.xscale("log")
    plt.xlim(1000,10000000)
    plt.ylabel("Relative Luminosity (A.U.)")
    plt.ylim(0,0.4)
    plt.title("Comparison of loss measures, $\epsilon = {:2.0f}$".format(epsilon[2]))
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/loss_eps{:2.0f}.png".format(epsilon[2]), dpi=DPI)
    plt.clf()

#%%
print("Big Guns")

conv_dynamic_aperture = {}
for epsilon in data:
	conv_dynamic_aperture[epsilon] = convolve_and_compute(data[epsilon], n_turns, n_angles_in_partition, stride)

conv_fit_parameters1 = {}    # fit1
conv_fit_parameters2 = {}    # fit2
conv_fit_parameters3 = {}    # fit3
conv_fit_parameters4 = {}    # fit4

best_conv_fit_parameters1 = {}    # fit1
best_conv_fit_parameters2 = {}    # fit2
best_conv_fit_parameters3 = {}    # fit3
best_conv_fit_parameters4 = {}    # fit4

print("fit 1")
for epsilon in conv_dynamic_aperture:
    temp = {}
    for angle in conv_dynamic_aperture[epsilon]:
        temp[angle] = non_linear_fit(conv_dynamic_aperture[epsilon][angle], n_turns, method = 1)
    conv_fit_parameters1[epsilon] = temp

for epsilon in conv_fit_parameters1:
    temp = {}
    for angle in conv_fit_parameters1[epsilon]:
        temp[angle] = select_best_fit(conv_fit_parameters1[epsilon][angle])
    best_conv_fit_parameters1[epsilon] = temp

print("fit 2")
for epsilon in conv_dynamic_aperture:
    temp = {}
    for angle in conv_dynamic_aperture[epsilon]:
        temp[angle] = non_linear_fit(conv_dynamic_aperture[epsilon][angle], n_turns, method = 2)
    conv_fit_parameters2[epsilon] = temp

for epsilon in conv_fit_parameters2:
    temp = {}
    for angle in conv_fit_parameters2[epsilon]:
        temp[angle] = select_best_fit(conv_fit_parameters2[epsilon][angle])
    best_conv_fit_parameters2[epsilon] = temp

print("fit 3")
for epsilon in conv_dynamic_aperture:
    temp = {}
    for angle in conv_dynamic_aperture[epsilon]:
        temp[angle] = non_linear_fit(conv_dynamic_aperture[epsilon][angle], n_turns, method = 3)
    conv_fit_parameters3[epsilon] = temp

for epsilon in conv_fit_parameters3:
    temp = {}
    for angle in conv_fit_parameters3[epsilon]:
        temp[angle] = select_best_fit(conv_fit_parameters3[epsilon][angle])
    best_conv_fit_parameters3[epsilon] = temp

print("fit 4")
for epsilon in conv_dynamic_aperture:
    temp = {}
    for angle in conv_dynamic_aperture[epsilon]:
        temp[angle] = non_linear_fit(conv_dynamic_aperture[epsilon][angle], n_turns, method = 4)
    conv_fit_parameters4[epsilon] = temp

for epsilon in conv_fit_parameters4:
    temp = {}
    for angle in conv_fit_parameters4[epsilon]:
        temp[angle] = select_best_fit(conv_fit_parameters4[epsilon][angle], True)
    best_conv_fit_parameters4[epsilon] = temp

#%%
print("Fit Parameter Comparison")

def plot_fit_comparison(nfit, fit_parameters, params_are_4 = False):
    for epsilon in fit_parameters:
        theta = []
        A = []
        B = []
        C = []
        k = []
        for angle in fit_parameters[epsilon]:
            theta.append(angle / np.pi)
            A.append(fit_parameters[epsilon][angle][0])
            B.append(fit_parameters[epsilon][angle][2])
            if params_are_4:
                C.append(fit_parameters[epsilon][angle][4])
                k.append(fit_parameters[epsilon][angle][6])
            else:
                k.append(fit_parameters[epsilon][angle][4])
        plt.plot(theta, A, marker = "o", linewidth = 0, label = "A")
        plt.plot(theta, B, marker = "*", linewidth = 0, label = "B")
        plt.plot(theta, k, marker = "^", linewidth = 0, label = "k")
        if params_are_4:
            plt.plot(theta, C, marker = "x", linewidth = 0, label = "C")
        plt.xlim((0,0.5))
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit values (A.U.)")
        plt.title("Fit {} values at different angles,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(nfit, epsilon[2], epsilon[0], epsilon[1]))
        plt.legend()
        plt.tight_layout()
        plt.savefig("img/many_angles{:}_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(nfit, epsilon[2], epsilon[0], epsilon[1]), dpi = DPI)
        plt.clf()

print("Comparison 1")

plot_fit_comparison(1, best_conv_fit_parameters1)

print("Comparison 2")

plot_fit_comparison(2, best_conv_fit_parameters2)

print("Comparison 3")

plot_fit_comparison(3, best_conv_fit_parameters3)

print("Comparison 4")

plot_fit_comparison(4, best_conv_fit_parameters4, True)

#%%
print("Draw 2D stability maps")

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
    plt.ylim(0,0.8)
    plt.grid(True)
    plt.title("Stable Region\n$(\omega_x = {:3.3f}, \omega_y = {:3.3f}, \epsilon = {:3.3f})$".format(key[0], key[1], key[2]))
    plt.tight_layout()
    plt.savefig("img/stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(key[2], key[0], key[1]), dpi = DPI)
    plt.clf()

#%%
print("Draw 2D stability maps from linscan")

stability_levels = np.array([1000, 10000, 100000, 1000000, 10000000])

for epsilon in lin_data:
    for level in stability_levels:
        x = []
        y = []
        x.append(0.)
        y.append(0.)
        i = -1
        for line in lin_data[epsilon]:
            i += 1
            j = 0
            while line[j] >= level:
                j += 1
            x.append((j - 1) * dx)
            y.append(i * dx)
        plt.fill(x, y, label="$turns = {}$".format(level))
    plt.legend()
    plt.xlabel("X coordinate (A.U.)")
    plt.ylabel("Y coordinate (A.U.)")
    plt.xlim(0,0.8)
    plt.ylim(0,0.8)
    plt.grid(True)
    plt.title("Stable Region (grid)\n$(\omega_x = {:3.3f}, \omega_y = {:3.3f}, \epsilon = {:3.3f})$".format(epsilon[0], epsilon[1], epsilon[2]))
    plt.tight_layout()
    plt.savefig("img/grid_stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(epsilon[2], epsilon[0], epsilon[1]), dpi = DPI)
    plt.clf()

#%%
# Concatenate
import cv2
for epsilon in lin_data:
    img1 = cv2.imread("img/grid_stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(epsilon[2], epsilon[0], epsilon[1]))
    img2 = cv2.imread("img/stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(epsilon[2], epsilon[0], epsilon[1]))
    vis = np.concatenate((img1,img2), axis=1)
    cv2.imwrite("img/concatenated_stability_eps{:2.0f}.png".format(epsilon[2]),vis)

#%%
# Convert to JPEG

converter.png_to_jpg("img/")
