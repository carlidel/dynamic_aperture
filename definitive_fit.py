# Includes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from scipy.optimize import curve_fit
from scipy import integrate

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

n_turns = np.array([1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 12000, 14000, 16000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 120000, 140000, 160000, 180000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000, 950000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000, 6500000, 7000000, 7500000, 8000000, 8500000, 9000000, 9500000, 10000000])

# Partition list for basic angle partitioning
partition_lists = [[0, np.pi / 2], # Always always keep this one 
                   #[0, np.pi / 4, np.pi / 2], 
                   #[0, np.pi / (2 * 3), np.pi / (3), np.pi / 2], 
                   #[0, np.pi / 8, np.pi * 2 / 8, np.pi * 3 / 8, np.pi / 2],
                   #[0, np.pi / 10, np.pi * 2 / 10, np.pi * 3 / 10, np.pi * 4 / 10, np.pi / 2],
                   #[0, np.pi / 12, np.pi * 2 / 12, np.pi * 3 / 12, np.pi * 4 / 12, np.pi * 5 / 12, np.pi / 2]
                  ]

# Exponential fit parameters
k_max = 5.
k_min = -5.
n_k = 200
k_possible_values = np.linspace(k_min, k_max, n_k)
k_error = k_possible_values[1] - k_possible_values[0]

#%%

# Function Definitions

def compute_D(contour_data, section_lenght, d_angle = dtheta):
    '''
    Given a list of distances from an angular scan
    and the total lenght of the section.
    Returns the defined Dynamic Aperture
    '''
    return integrate.simps(contour_data, dx=d_angle) / (section_lenght)

def compute_D_error(contour_data, d_lenght = dx, d_angle = dtheta):
    '''
    Given the list of distances from an angular scan,
    computes the error estimation
    '''
    first_derivatives = np.asarray([(contour_data[k+1] - contour_data[k]) / d_angle for k in range(len(contour_data) - 1)])
    average_derivative = np.average(first_derivatives)
    return (np.sqrt(d_lenght * d_lenght / 4 + average_derivative * average_derivative * d_angle * d_angle / 4))

def divide_and_compute(data, n_turns, partition_list = [0, np.pi/2], d_lenght = dx, d_angle = dtheta):
    '''
    data is a dictionary containing the simulation data.
    n_turns is an array of the times to explore.
    partition_list is the list of partition to analyze separately.
    Returns a dictionary per angles analyzed
    '''
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
            D[t] = compute_D(limit, partition_list[i+1] - partition_list[i], d_angle)
            Err[t] = compute_D_error(limit, d_lenght, d_angle)           
        angle[(partition_list[i] + partition_list[i + 1]) / 2] = (D, Err)
    return angle

def FIT1(x, A, B, k):
    return A + B / np.log10(x) ** k

def FIT2(x, A, B, k):
    return B / (np.log10(x) - A) ** k

def non_linear_fit(data, err_data, n_turns, method = 1, k_possible_values = k_possible_values, k_error = k_error):
    '''
    data is a dictionary of the dynamic aperture data with n_turns as keys
    err_data is corrispective data with n_turns as keys
    '''
    fit1 = lambda x, A, B : A + B / np.log10(x) ** k
    chi1 = lambda x, y, sigma, popt : (1 / (len(n_turns) - 3)) * np.sum(((y - fit1(x, popt[0], popt[1])) / sigma)**2)
    
    fit2 = lambda x, A, B : B / (np.log10(x) - A) ** k   
    chi2 = lambda x, y, sigma, popt : (1 / (len(n_turns) - 3)) * np.sum(((y - fit2(x, popt[0], popt[1])) / sigma)**2)

    if method == 1:
        explore_k = {}
        for number in k_possible_values:
            k = number
            try:
                popt, pcov = curve_fit(fit1, n_turns, [data[i] for i in n_turns], p0=[0.,1.], sigma=[err_data[i] for i in n_turns])
                explore_k[k] = (popt, pcov, (chi1(n_turns, [data[i] for i in n_turns], [err_data[i] for i in n_turns], popt)))
            except RuntimeError:
                pass
        assert len(explore_k) > 0
        return explore_k
    
    elif method == 2:
        explore_k = {}
        # FACT: IT DOES NOT CONVERGE WITH k < 0!!! JUST WASTED TIME..
        for number in k_possible_values:
            k = number
            try:
                popt, pcov = curve_fit(fit2, n_turns, [data[i] for i in n_turns], p0=[0.,1.], sigma=[err_data[i] for i in n_turns])
                explore_k[k] = (popt, pcov, chi2(n_turns, [data[i] for i in n_turns], [err_data[i] for i in n_turns], popt))
            except RuntimeError:
                pass
        assert len(explore_k) > 0
        return explore_k
    
    else:
        print("No method")
        assert False

def select_best_fit(parameters):
    best = sorted(parameters.items(), key = lambda kv: kv[1][2])[0]
    return(best[1][0][0], np.sqrt(best[1][1][0][0]), best[1][0][1], np.sqrt(best[1][1][1][1]), best[0])

#%%
print("load data")

data = pickle.load(open("radscan_dx01_firstonly_dictionary.pkl", "rb"))
lin_data = pickle.load(open("linscan_dx01_firstonly_dictionary.pkl", "rb"))

#%%
dynamic_aperture = {}

for epsilon in sorted(data):
    dyn_temp = {}
    for partition_list in partition_lists:
        dyn_temp[len(partition_list)-1] = divide_and_compute(data[epsilon], n_turns, partition_list)
    dynamic_aperture[epsilon] = dyn_temp

#%%
print("Fit on Partitions")

fit_parameters1 = {}   
fit_parameters2 = {}
best_fit_parameters1 = {}
best_fit_parameters2 = {}

for epsilon in dynamic_aperture:
    print(epsilon)
    # fit1
    fit_parameters_epsilon = {}
    best_fit_parameters_epsilon = {}

    for partition_list in partition_lists:
        fit = {}
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list)-1]:
            fit[angle] = non_linear_fit(dynamic_aperture[epsilon][len(partition_list)-1][angle][0], dynamic_aperture[epsilon][len(partition_list)-1][angle][1], n_turns, method=1)
            best[angle] = select_best_fit(fit[angle])
        fit_parameters_epsilon[len(partition_list)-1] = fit
        best_fit_parameters_epsilon[len(partition_list)-1] = best

    fit_parameters1[epsilon] = fit_parameters_epsilon
    best_fit_parameters1[epsilon] = best_fit_parameters_epsilon

    # fit2
    fit_parameters_epsilon = {}
    best_fit_parameters_epsilon = {}

    for partition_list in partition_lists:
        fit = {}
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list)-1]:
            fit[angle] = non_linear_fit(dynamic_aperture[epsilon][len(partition_list)-1][angle][0], dynamic_aperture[epsilon][len(partition_list)-1][angle][1], n_turns, method=2)
            best[angle] = select_best_fit(fit[angle])
        fit_parameters_epsilon[len(partition_list)-1] = fit
        best_fit_parameters_epsilon[len(partition_list)-1] = best

    fit_parameters2[epsilon] = fit_parameters_epsilon
    best_fit_parameters2[epsilon] = best_fit_parameters_epsilon

#%%
print("Plot fits")

def plot_fit_basic(numfit, best_fit, N, epsilon, angle, n_turns, dynamic_aperture, func, y_bar = False):
    plt.errorbar(n_turns, [dynamic_aperture[epsilon][N][angle][0][i] for i in n_turns], yerr=[dynamic_aperture[epsilon][N][angle][1][i] for i in n_turns], linewidth = 0, elinewidth = 2, label = 'Data')
    plt.plot(n_turns, func(n_turns, best_fit[0], best_fit[2], best_fit[4]), 'g--', linewidth=0.5, label = 'fit: $A={:6.3f}, B={:6.3f}, k={:6.3f}$'.format(best_fit[0], best_fit[2], best_fit[4]))
    if y_bar:
        plt.axhline(y = best_fit[0], color = 'r', linestyle = '-', label = '$y=A={:6.3f}$'.format(best_fit[0]))
    plt.xlabel("$N$ turns")
    plt.xscale("log")
    plt.ylabel("$D (A.U.)$")
    plt.ylim(0.,1.)
    plt.title("Fit formula {},\n$dx = {:2.2f}, dth = {:3.2f}, mid\,angle = {:3.3f}$,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".format(numfit, dx, dtheta, angle, epsilon[2], epsilon[0], epsilon[1]))
    # Tweak for legend.
    plt.plot([],[],'', linewidth=0, label = "$A = {:.2} \pm {:.2}$".format(best_fit[0], best_fit[1]))
    plt.plot([],[],'', linewidth=0, label = "$B = {:.2} \pm {:.2}$".format(best_fit[2], best_fit[3]))
    plt.plot([],[],'', linewidth=0, label = "$k = {:.2} \pm {:.2}$".format(best_fit[4], k_error))
    # And then the legend.
    plt.legend(prop={"size" : 7})
    plt.tight_layout()
    plt.savefig("img/fit{}_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}_angle{:3.3f}_Npart{}.png".format(numfit, epsilon[2], epsilon[0], epsilon[1],angle,1), dpi = DPI) ## for now partition list is set to 1
    plt.clf()

print("Plot fit 1.")
for epsilon in best_fit_parameters1:
    for partition in best_fit_parameters1[epsilon]:
        for angle in best_fit_parameters1[epsilon][partition]:
            plot_fit_basic(1, best_fit_parameters1[epsilon][partition][angle], partition, epsilon, angle, n_turns, dynamic_aperture, FIT1, True)

print("Plot fit 2.")
for epsilon in best_fit_parameters2:
    for partition in best_fit_parameters2[epsilon]:
        for angle in best_fit_parameters2[epsilon][partition]:
            plot_fit_basic(2, best_fit_parameters2[epsilon][partition][angle], partition, epsilon, angle, n_turns, dynamic_aperture, FIT2, True)

#%%
print("Plot Fit Performances")

def compare_fit_chi_squared(fit1, fit2, epsilon, n_partitions = 1, angle = np.pi/4):
    plt.plot(list(fit1.keys()), [x[2] for x in list(fit1.values())], marker = "o", markersize = 0.5,  linewidth = 0.5, label = "fit1")
    plt.plot(list(fit2.keys()), [x[2] for x in list(fit2.values())], marker = "o", markersize = 0.5,  linewidth = 0.5, label = "fit2")
    plt.xlabel("k value")
    plt.ylabel("Chi-Squared value")
    plt.title("Fit Performance Comparison (Chi-Squared based), $\epsilon = {:2.0f}$,\n number of partitions $= {}$, central angle $= {:2.2f}$".format(epsilon, n_partitions, angle))
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/fit_performance_comparison_eps{:2.0f}_npart{}_central{:2.2f}.png".format(epsilon, n_partitions, angle), dpi = DPI)
    plt.clf()

for epsilon in fit_parameters1:
    for N in fit_parameters1[epsilon]:
        for angle in fit_parameters1[epsilon][N]:
            compare_fit_chi_squared(fit_parameters1[epsilon][N][angle], fit_parameters2[epsilon][N][angle], epsilon[2], N, angle)

#%%
print("Is This Loss?")

# Sigmas for gaussian distribution to explore
sigmas = [0.25, 0.5, 0.75, 1, 1.25, 1.50, 2.]

# Functions
def intensity_zero_gaussian(x, y, sigma_x = 1, sigma_y = 1):
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-((x**2/(2*sigma_x**2))+(y**2/(2*sigma_y**2))))

def relative_intensity_D_law(D, sigma = 1): # Assuming equal sigma for 2Ds
    return 1 - np.exp(- D ** 2 / (2 * sigma * sigma))

def grid_intensity(grid, dx = dx):
    # TODO :: IMPROVE?
    return integrate.trapz(integrate.simps(grid, dx = dx), dx = dx)

def multiple_partition_intensity(best_fit_params, fit_func, n_parts, time, sigma):
    # Let's treat it as a basic summatory
    intensity = 0.
    for angle in best_fit_params:
        current_dynamic_aperture = fit_func(time, best_fit_params[angle][0], best_fit_params[angle][2], best_fit_params[angle][4])
        intensity += relative_intensity_D_law(current_dynamic_aperture, sigma) / n_parts
    return intensity

# Weights at beginning
loss_precise = {}
loss_D_fit1 = {}
loss_D_fit2 = {}

for sigma in sigmas:
    print(sigma)
    weights = np.array([[intensity_zero_gaussian(x * dx, y * dx, sigma, sigma) for x in range(80)] for y in range(80)])
    I0 = grid_intensity(np.array([[intensity_zero_gaussian(x * dx, y * dx, sigma, sigma) for x in range(1000)] for y in range(1000)]))

    print("precise")

    loss_precise_temp = {}
    for epsilon in lin_data:
        print(epsilon)
        intensity_evolution = [0.25]
        for time in n_turns:
            mask = np.copy(lin_data[epsilon])
            mask[mask < time] = 0
            mask[mask >= time] = 1
            masked_weights = weights * mask
            intensity_evolution.append(grid_intensity(masked_weights))
        loss_precise_temp[epsilon] = np.asarray(intensity_evolution) / I0
    loss_precise[sigma] = loss_precise_temp

    print("from fit1")

    loss_D_fit_temp = {}
    for epsilon in best_fit_parameters1:
        print(epsilon)
        loss_D_fit_temp_part = {}
        for N in best_fit_parameters1[epsilon]:
            intensity_evolution = [1.]
            for time in n_turns:
                intensity_evolution.append(multiple_partition_intensity(best_fit_parameters1[epsilon][N], FIT1, N, time, sigma))
            loss_D_fit_temp_part[N] = np.asarray(intensity_evolution)
            print(loss_D_fit_temp_part[N])
        loss_D_fit_temp[epsilon] = loss_D_fit_temp_part
    loss_D_fit1[sigma] = loss_D_fit_temp
    
    print("from fit2")

    loss_D_fit_temp = {}
    for epsilon in best_fit_parameters2:
        print(epsilon)
        loss_D_fit_temp_part = {}
        for N in best_fit_parameters2[epsilon]:
            intensity_evolution = [1.]
            for time in n_turns:
                intensity_evolution.append(multiple_partition_intensity(best_fit_parameters2[epsilon][N], FIT2, N, time, sigma))
            loss_D_fit_temp_part[N] = np.asarray(intensity_evolution)
        loss_D_fit_temp[epsilon] = loss_D_fit_temp_part
    loss_D_fit2[sigma] = loss_D_fit_temp

#%%
print("Plot both loss fits.")

for sigma in sigmas:
    print(sigma)
    for epsilon in loss_D_fit1[sigma]:
        for N in loss_D_fit1[sigma][epsilon]:
            plt.plot(np.concatenate((np.array([0]),n_turns))[1:], loss_precise[sigma][epsilon][1:], linewidth=0.5, label="Precise loss")
            plt.plot(np.concatenate((np.array([0]),n_turns))[1:], loss_D_fit1[sigma][epsilon][N][1:], linewidth=0.5, label="D loss FIT1")
            plt.plot(np.concatenate((np.array([0]),n_turns))[1:], np.absolute(loss_precise[sigma][epsilon] - loss_D_fit1[sigma][epsilon][N])[1:], linewidth=0.5, label="Difference FIT1")
            plt.plot(np.concatenate((np.array([0]),n_turns))[1:], loss_D_fit2[sigma][epsilon][N][1:], linewidth=0.5, label="D loss FIT2")
            plt.plot(np.concatenate((np.array([0]),n_turns))[1:], np.absolute(loss_precise[sigma][epsilon] - loss_D_fit2[sigma][epsilon][N])[1:], linewidth=0.5, label="Difference FIT2")
            plt.xlabel("N turns")
            plt.xscale("log")
            plt.xlim(1e3,1e7)
            plt.ylabel("Relative Luminosity")
            plt.ylim(bottom=0)
            plt.title("Comparison of loss measures (FIT1 and FIT2), $\sigma = {:2.1f}$, $\epsilon = {:2.0f}$, $N parts = {}$".format(sigma,epsilon[2], N))
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("img/loss_both_sig{:2.1f}_eps{:2.0f}_npart{}.png".format(sigma,epsilon[2], N), dpi=DPI)
            plt.clf()

#%%
print("Draw 2D stability maps")

stability_levels = np.array([1000, 10000, 100000, 1000000, 10000000])

for key in data:
    fig, ax = plt.subplots()
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
        ax.fill(x, y, label="$turns = 10^{}$".format(int(np.log10(level))))
    ax.legend()
    ax.set_xlabel("X coordinate (A.U.)")
    ax.set_ylabel("Y coordinate (A.U.)")
    ax.set_xlim(0,0.8)
    ax.set_ylim(0,0.8)
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.set_title("Stable Region (angular scan)\n$(\omega_x = {:3.3f}, \omega_y = {:3.3f}, \epsilon = {:3.3f})$".format(key[0], key[1], key[2]))
    fig.tight_layout()
    fig.savefig("img/stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(key[2], key[0], key[1]), dpi = DPI)
    plt.clf()

#%%
print("Draw 2D stability maps from linscan")

from matplotlib.colors import LogNorm

for epsilon in lin_data:
    temp = np.copy(lin_data[epsilon])
    temp += 1
    plt.imshow(temp, origin="lower", extent=(0,0.8,0,0.8), norm=LogNorm(vmin=1, vmax=10000000))
    plt.xlabel("X coordinate (A.U.)")
    plt.ylabel("Y coordinate (A.U.)")
    plt.xlim(0,0.8)
    plt.ylim(0,0.8)
    plt.grid(True)
    plt.title("Stable Region (grid scan), number of turns\n$(\omega_x = {:3.3f}, \omega_y = {:3.3f}, \epsilon = {:3.3f})$".format(epsilon[0], epsilon[1], epsilon[2]))
    plt.colorbar()
    #plt.tight_layout()
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
# Compare every region

from matplotlib.colors import LogNorm

stability_levels = np.array([10000000, 5000000, 1000000, 500000, 100000, 50000, 10000, 5000, 1000])    
    
for epsilon in lin_data:
    for level in stability_levels:
        fig, ax = plt.subplots()
        temp = np.copy(lin_data[epsilon])
        temp += 1
        coso = ax.imshow(temp, origin="lower", extent=(0,0.8,0,0.8), norm=LogNorm(vmin=level, vmax=10000000))
        x = []
        y = []
        x.append(0.)
        y.append(0.)
        for line in sorted(data[epsilon]):
            j = 0
            while data[epsilon][line][j] >= level:
                j += 1
            x.append((j - 1) * dx * np.cos(line))
            y.append((j - 1) * dx * np.sin(line))
        ax.fill(x, y, "r",label="Angle scan".format(int(np.log10(level))))
        ax.legend()
        ax.set_xlabel("X coordinate (A.U.)")
        ax.set_ylabel("Y coordinate (A.U.)")
        ax.set_xlim(0,0.8)
        ax.set_ylim(0,0.8)
        ax.set_aspect("equal", "box")
        ax.grid(True)
        fig.colorbar(coso, ax=ax)
        ax.set_title("Comparision of scans, $\epsilon = {:2.0f}$, $N = {}$".format(epsilon[2], level))
        fig.savefig("img/comparison_eps{:2.0f}_N{}.png".format(epsilon[2], level), dpi = DPI)
        plt.close()

#%%
# Convert to JPEG

converter.png_to_jpg("img/")
