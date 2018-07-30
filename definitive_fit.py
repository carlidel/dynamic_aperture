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

n_turns = np.array([
    1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
    5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 12000, 14000,
    16000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000,
    60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 120000,
    140000, 160000, 180000, 200000, 250000, 300000, 350000, 400000, 450000,
    500000, 550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000,
    950000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2500000,
    3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000, 6500000,
    7000000, 7500000, 8000000, 8500000, 9000000, 9500000, 10000000
])

# Partition list for basic angle partitioning
partition_lists = [
    [0, np.pi / 2],  # Always always keep this one 
    [0, np.pi / 4, np.pi / 2],
    [0, np.pi / (2 * 3), np.pi / (3), np.pi / 2],
    [0, np.pi / 8, np.pi * 2 / 8, np.pi * 3 / 8, np.pi / 2],
    [0, np.pi / 10, np.pi * 2 / 10, np.pi * 3 / 10, np.pi * 4 / 10, np.pi / 2],
    [
        0, np.pi / 12, np.pi * 2 / 12, np.pi * 3 / 12, np.pi * 4 / 12,
        np.pi * 5 / 12, np.pi / 2
    ]
]

# Exponential fit parameters
k_max = 10.
k_min = -10.
n_k = 401
k_possible_values = np.linspace(k_min, k_max, n_k)
k_error = (k_possible_values[1] - k_possible_values[0]) / 2

#%%

# Function Definitions


def compute_D(contour_data, section_lenght, d_angle=dtheta):
    '''
    Given a list of distances from an angular scan
    and the total lenght of the section.
    Returns the defined Dynamic Aperture
    '''
    #return integrate.simps(contour_data, dx=d_angle) / (section_lenght)
    return np.average(contour_data)


def compute_D_error(contour_data, d_lenght=dx, d_angle=dtheta):
    '''
    Given the list of distances from an angular scan,
    computes the error estimation
    '''
    first_derivatives = np.asarray(
        [(contour_data[k + 1] - contour_data[k]) / d_angle
         for k in range(len(contour_data) - 1)])
    average_derivative = np.average(first_derivatives)
    return (np.sqrt(d_lenght * d_lenght / 4 + average_derivative *
                    average_derivative * d_angle * d_angle / 4))


def divide_and_compute(data,
                       n_turns,
                       partition_list=[0, np.pi / 2],
                       d_lenght=dx,
                       d_angle=dtheta):
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
                if line >= partition_list[i] and line <= partition_list[i + 1]:
                    n_angles += 1
                    j = 0
                    while data[line][j] >= t:
                        j += 1
                    limit.append((j - 1) * dx)
            assert n_angles >= 5
            limit = np.asarray(limit)
            D[t] = compute_D(limit, partition_list[i + 1] - partition_list[i],
                             d_angle)
            Err[t] = compute_D_error(limit, d_lenght, d_angle)
        angle[(partition_list[i] + partition_list[i + 1]) / 2] = (D, Err)
    return angle


def make_countour_data(data, n_turns):
    angle = {}
    for theta in data:
        temp = {}
        for time in n_turns:
            j = 0
            while data[theta][j] >= time:
                j += 1
            temp[time] = ((j - 1) * dx)
        angle[theta] = temp
    return angle


def FIT1(x, A, B, k):
    return A + B / np.log10(x)**k


def FIT2(x, A, B, k):
    return B / (np.log10(x) - A)**k


def non_linear_fit(data,
                   err_data,
                   n_turns,
                   method=1,
                   k_possible_values=k_possible_values,
                   k_error=k_error):
    '''
    data is a dictionary of the dynamic aperture data with n_turns as keys
    err_data is corrispective data with n_turns as keys
    '''
    fit1 = lambda x, A, B: A + B / np.log10(x)**k
    chi1 = lambda x, y, sigma, popt : (1 / (len(n_turns) - 3)) * np.sum(((y - fit1(x, popt[0], popt[1])) / sigma)**2)

    fit2 = lambda x, A, B: B / (np.log10(x) - A)**k
    chi2 = lambda x, y, sigma, popt : (1 / (len(n_turns) - 3)) * np.sum(((y - fit2(x, popt[0], popt[1])) / sigma)**2)

    if method == 1:
        explore_k = {}
        for number in k_possible_values:
            k = number
            try:
                popt, pcov = curve_fit(
                    fit1,
                    n_turns, [data[i] for i in n_turns],
                    p0=[0., 1.],
                    sigma=[err_data[i] for i in n_turns])
                explore_k[k] = (popt, pcov,
                                (chi1(n_turns, [data[i] for i in n_turns],
                                      [err_data[i] for i in n_turns], popt)))
            except RuntimeError:
                pass
        assert len(explore_k) > 0
        return explore_k

    elif method == 2:
        explore_k = {}
        # FACT: IT DOES NOT CONVERGE WITH k < 0!!! JUST WASTED TIME...
        for number in k_possible_values:
            if number >= 0:
                k = number
                try:
                    popt, pcov = curve_fit(
                        fit2,
                        n_turns, [data[i] for i in n_turns],
                        p0=[0., 1.],
                        sigma=[err_data[i] for i in n_turns])
                    explore_k[k] = (popt, pcov,
                                    chi2(n_turns, [data[i] for i in n_turns],
                                         [err_data[i] for i in n_turns], popt))
                except RuntimeError:
                    pass
        assert len(explore_k) > 0
        return explore_k

    else:
        print("No method")
        assert False


def select_best_fit(parameters):
    best = sorted(parameters.items(), key=lambda kv: kv[1][2])[0]
    return (best[1][0][0], np.sqrt(best[1][1][0][0]), best[1][0][1],
            np.sqrt(best[1][1][1][1]), best[0])


#%%
print("load data")

data = pickle.load(open("radscan_dx01_firstonly_dictionary.pkl", "rb"))
lin_data = pickle.load(open("linscan_dx01_firstonly_dictionary.pkl", "rb"))

contour_data = {}
for epsilon in data:
    contour_data[epsilon] = make_countour_data(data[epsilon], n_turns)

#%%
dynamic_aperture = {}

for epsilon in sorted(data):
    dyn_temp = {}
    for partition_list in partition_lists:
        dyn_temp[len(partition_list) - 1] = divide_and_compute(
            data[epsilon], n_turns, partition_list)
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
        for angle in dynamic_aperture[epsilon][len(partition_list) - 1]:
            fit[angle] = non_linear_fit(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns,
                method=1)
            best[angle] = select_best_fit(fit[angle])
        fit_parameters_epsilon[len(partition_list) - 1] = fit
        best_fit_parameters_epsilon[len(partition_list) - 1] = best

    fit_parameters1[epsilon] = fit_parameters_epsilon
    best_fit_parameters1[epsilon] = best_fit_parameters_epsilon

    # fit2
    fit_parameters_epsilon = {}
    best_fit_parameters_epsilon = {}

    for partition_list in partition_lists:
        fit = {}
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list) - 1]:
            fit[angle] = non_linear_fit(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns,
                method=2)
            best[angle] = select_best_fit(fit[angle])
        fit_parameters_epsilon[len(partition_list) - 1] = fit
        best_fit_parameters_epsilon[len(partition_list) - 1] = best

    fit_parameters2[epsilon] = fit_parameters_epsilon
    best_fit_parameters2[epsilon] = best_fit_parameters_epsilon

#%%
print("Plot fits")


def plot_fit_basic(numfit,
                   best_fit,
                   N,
                   epsilon,
                   angle,
                   n_turns,
                   dynamic_aperture,
                   func,
                   y_bar=False):
    plt.errorbar(
        n_turns, [dynamic_aperture[epsilon][N][angle][0][i] for i in n_turns],
        yerr=[dynamic_aperture[epsilon][N][angle][1][i] for i in n_turns],
        linewidth=0,
        elinewidth=2,
        label='Data')
    plt.plot(
        n_turns,
        func(n_turns, best_fit[0], best_fit[2], best_fit[4]),
        'g--',
        linewidth=0.5,
        label='fit: $A={:6.3f}, B={:6.3f}, k={:6.3f}$'.format(
            best_fit[0], best_fit[2], best_fit[4]))
    if y_bar:
        plt.axhline(
            y=best_fit[0],
            color='r',
            linestyle='-',
            label='$y=A={:6.3f}$'.format(best_fit[0]))
    plt.xlabel("$N$ turns")
    plt.xscale("log")
    plt.ylabel("$D (A.U.)$")
    plt.ylim(0., 1.)
    plt.title(
        "Fit formula {},\n$dx = {:2.2f}, dth = {:3.2f}, mid\,angle = {:3.3f}$,\n$\epsilon = {:2.0f}, \omega_x = {:3.3f}, \omega_y = {:3.3f}$".
        format(numfit, dx, dtheta, angle, epsilon[2], epsilon[0], epsilon[1]))
    # Tweak for legend.
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$A = {:.2} \pm {:.2}$".format(best_fit[0], best_fit[1]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$B = {:.2} \pm {:.2}$".format(best_fit[2], best_fit[3]))
    plt.plot(
        [], [],
        '',
        linewidth=0,
        label="$k = {:.2} \pm {:.2}$".format(best_fit[4], k_error))
    # And then the legend.
    plt.legend(prop={"size": 7})
    plt.tight_layout()
    plt.savefig(
        "img/fit{}_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}_angle{:3.3f}_Npart{}.png".
        format(numfit, epsilon[2], epsilon[0], epsilon[1], angle, N),
        dpi=DPI)  ## for now partition list is set to 1
    plt.clf()


print("Plot fit 1.")
for epsilon in best_fit_parameters1:
    for partition in best_fit_parameters1[epsilon]:
        for angle in best_fit_parameters1[epsilon][partition]:
            plot_fit_basic(1, best_fit_parameters1[epsilon][partition][angle],
                           partition, epsilon, angle, n_turns,
                           dynamic_aperture, FIT1, True)

print("Plot fit 2.")
for epsilon in best_fit_parameters2:
    for partition in best_fit_parameters2[epsilon]:
        for angle in best_fit_parameters2[epsilon][partition]:
            plot_fit_basic(2, best_fit_parameters2[epsilon][partition][angle],
                           partition, epsilon, angle, n_turns,
                           dynamic_aperture, FIT2, True)

#%%
print("Fit parameters evolution.")


def fit_parameters_evolution(fit_parameters,
                             title="plot",
                             namefile="plot",
                             method=1):
    theta = []
    A = []
    A_err = []
    B = []
    B_err = []
    k = []
    k_err = []
    for N in fit_parameters:
        theta_temp = []
        A_temp = []
        B_temp = []
        k_temp = []
        A_temp_err = []
        B_temp_err = []
        k_temp_err = []
        for angle in fit_parameters[N]:
            theta_temp.append(angle / np.pi)
            A_temp.append(fit_parameters[N][angle][0])
            B_temp.append(fit_parameters[N][angle][2])
            k_temp.append(fit_parameters[N][angle][4])
            A_temp_err.append(fit_parameters[N][angle][1])
            B_temp_err.append(fit_parameters[N][angle][3])
            k_temp_err.append(k_error)
        theta.append(theta_temp)
        A.append(A_temp)
        B.append(B_temp)
        k.append(k_temp)
        A_err.append(A_temp_err)
        B_err.append(B_temp_err)
        k_err.append(k_temp_err)
    #print(A)
    #print(B)
    for i in range(len(A)):
        plt.errorbar(
            theta[i],
            A[i],
            yerr=A_err[i],
            xerr=(0.25 / len(A[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value " + ("A" if method == 2 else "$D_\infty$ ") +
                   " (A.U.)")
        plt.title(title + ", " + ("A " if method == 2 else "$D_\infty$ ") +
                  "parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig(
        "img/" + namefile + ("_A.png" if method == 2 else "_Dinf.png"),
        dpi=DPI)
    plt.clf()
    for i in range(len(B)):
        plt.errorbar(
            theta[i],
            B[i],
            yerr=B_err[i],
            xerr=(0.25 / len(B[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value B (A.U.)")
        plt.title(title + ", B parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig("img/" + namefile + "_B.png", dpi=DPI)
    plt.clf()
    for i in range(len(k)):
        plt.errorbar(
            theta[i],
            k[i],
            yerr=k_err[i],
            xerr=(0.25 / len(k[i])),
            linewidth=0,
            elinewidth=1)
        plt.xlabel("Theta $(rad / \pi)$")
        plt.ylabel("Fit value k (A.U.)")
        plt.title(title + ", k parameter")
        plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        plt.tight_layout()
    plt.savefig("img/" + namefile + "_k.png", dpi=DPI)
    plt.clf()


for epsilon in best_fit_parameters1:
    fit_parameters_evolution(
        best_fit_parameters1[epsilon],
        "FIT1, $\epsilon = {:2.0f}$".format(epsilon[2]),
        "comp_fit1_eps{:2.0f}".format(epsilon[2]),
        method=1)

for epsilon in best_fit_parameters2:
    fit_parameters_evolution(
        best_fit_parameters2[epsilon],
        "FIT2, $\epsilon = {:2.0f}$".format(epsilon[2]),
        "comp_fit2_eps{:2.0f}".format(epsilon[2]),
        method=2)

#%%
print("Plot Fit Performances")


def compare_fit_chi_squared(fit1,
                            fit2,
                            epsilon,
                            n_partitions=1,
                            angle=np.pi / 4):
    plt.plot(
        list(fit1.keys()), [x[2] for x in list(fit1.values())],
        marker="o",
        markersize=0.5,
        linewidth=0.5,
        label="fit1")
    plt.plot(
        list(fit2.keys()), [x[2] for x in list(fit2.values())],
        marker="o",
        markersize=0.5,
        linewidth=0.5,
        label="fit2")
    plt.xlabel("k value")
    plt.ylabel("Chi-Squared value")
    plt.title(
        "Fit Performance Comparison (Chi-Squared based), $\epsilon = {:2.0f}$,\n number of partitions $= {}$, central angle $= {:2.2f}$".
        format(epsilon, n_partitions, angle))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "img/fit_performance_comparison_eps{:2.0f}_npart{}_central{:2.2f}.png".
        format(epsilon, n_partitions, angle),
        dpi=DPI)
    plt.clf()


for epsilon in fit_parameters1:
    for N in fit_parameters1[epsilon]:
        for angle in fit_parameters1[epsilon][N]:
            compare_fit_chi_squared(fit_parameters1[epsilon][N][angle],
                                    fit_parameters2[epsilon][N][angle],
                                    epsilon[2], N, angle)

#%%
print("Is This Loss?")

# Sigmas for gaussian distribution to explore
sigmas = [0.2, 0.25, 0.5, 0.75, 1]


# Functions
def intensity_zero_gaussian(x, y, sigma_x=1, sigma_y=1):
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-(
        (x**2 / (2 * sigma_x**2)) + (y**2 / (2 * sigma_y**2))))


def relative_intensity_D_law(D, sigma=1):  # Assuming equal sigma for 2Ds
    return 1 - np.exp(-D**2 / (2 * sigma * sigma))


def D_from_loss(loss, sigma=1):  # INVERSE FORMULA
    return np.sqrt(-(np.log(-(loss - 1))) * (2 * sigma * sigma))


def grid_intensity(grid, dx=dx):
    # TODO :: IMPROVE?
    return integrate.simps(integrate.simps(grid, dx=dx), dx=dx)


def single_partition_intensity(best_fit_params, fit_func, time, sigma):
    current_dynamic_aperture = fit_func(time, best_fit_params[0],
                                        best_fit_params[2], best_fit_params[4])
    return relative_intensity_D_law(current_dynamic_aperture, sigma)


def multiple_partition_intensity(best_fit_params, fit_func, n_parts, time,
                                 sigma):
    # Let's treat it as a basic summatory
    intensity = 0.
    for angle in best_fit_params:
        current_dynamic_aperture = fit_func(time, best_fit_params[angle][0],
                                            best_fit_params[angle][2],
                                            best_fit_params[angle][4])
        intensity += relative_intensity_D_law(current_dynamic_aperture,
                                              sigma) / n_parts
    return intensity


def error_loss_estimation(best_fit_params, fit_func, contour_data, n_parts,
                          time, sigma):
    error = 0.
    for angle in best_fit_params:
        current_dynamic_aperture = fit_func(time, best_fit_params[angle][0],
                                            best_fit_params[angle][2],
                                            best_fit_params[angle][4])
        error_list = []
        angle_list = []
        for theta in contour_data:
            if angle - (np.pi /
                        (n_parts * 2)) <= theta <= angle + (np.pi /
                                                            (n_parts * 2)):
                error_list.append(
                    np.absolute(current_dynamic_aperture -
                                contour_data[theta][time]))
                angle_list.append(theta)
        error_list = np.asarray(error_list)
        angle_list = np.asarray(angle_list)
        error += (2 / np.pi) * np.exp(
            -(current_dynamic_aperture**2) /
            (2 * sigma * sigma)) * current_dynamic_aperture * integrate.simps(
                error_list, x=angle_list)
    return error


#%%
# Weights at beginning
loss_precise = {}
loss_D_fit1 = {}
loss_D_fit2 = {}
loss_D_fit1_err = {}
loss_D_fit2_err = {}

for sigma in sigmas:
    print(sigma)
    weights = np.array([[
        intensity_zero_gaussian(x * dx, y * dx, sigma, sigma)
        for x in range(80)
    ] for y in range(80)])
    I0 = grid_intensity(
        np.array([[
            intensity_zero_gaussian(x * dx, y * dx, sigma, sigma)
            for x in range(1000)
        ] for y in range(1000)]))

    print("precise")

    loss_precise_temp = {}
    for epsilon in lin_data:
        print(epsilon)
        intensity_evolution = [I0]
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
    loss_D_fit_temp_err = {}
    for epsilon in best_fit_parameters1:
        print(epsilon)
        loss_D_fit_temp_part = {}
        loss_D_fit_temp_part_err = {}
        for N in best_fit_parameters1[epsilon]:
            intensity_evolution = [1.]
            error_evolution = [0]
            for time in n_turns:
                intensity_evolution.append(
                    multiple_partition_intensity(
                        best_fit_parameters1[epsilon][N], FIT1, N, time,
                        sigma))
                error_evolution.append(
                    error_loss_estimation(best_fit_parameters1[epsilon][N],
                                          FIT1, contour_data[epsilon], N, time,
                                          sigma))
            loss_D_fit_temp_part[N] = np.asarray(intensity_evolution)
            loss_D_fit_temp_part_err[N] = np.asarray(error_evolution)
            #print(loss_D_fit_temp_part[N])
        loss_D_fit_temp[epsilon] = loss_D_fit_temp_part
        loss_D_fit_temp_err[epsilon] = loss_D_fit_temp_part_err
    loss_D_fit1[sigma] = loss_D_fit_temp
    loss_D_fit1_err[sigma] = loss_D_fit_temp_err

    print("from fit2")

    loss_D_fit_temp = {}
    loss_D_fit_temp_err = {}
    for epsilon in best_fit_parameters2:
        print(epsilon)
        loss_D_fit_temp_part = {}
        loss_D_fit_temp_part_err = {}
        for N in best_fit_parameters2[epsilon]:
            intensity_evolution = [1.]
            error_evolution = [0]
            for time in n_turns:
                intensity_evolution.append(
                    multiple_partition_intensity(
                        best_fit_parameters2[epsilon][N], FIT2, N, time,
                        sigma))
                error_evolution.append(
                    error_loss_estimation(best_fit_parameters2[epsilon][N],
                                          FIT2, contour_data[epsilon], N, time,
                                          sigma))
            loss_D_fit_temp_part[N] = np.asarray(intensity_evolution)
            loss_D_fit_temp_part_err[N] = np.asarray(error_evolution)
            #print(loss_D_fit_temp_part[N])
        loss_D_fit_temp[epsilon] = loss_D_fit_temp_part
        loss_D_fit_temp_err[epsilon] = loss_D_fit_temp_part_err
    loss_D_fit2[sigma] = loss_D_fit_temp
    loss_D_fit2_err[sigma] = loss_D_fit_temp_err

#%%
print("Fit on precise loss.")

fit_precise_loss1 = {}
fit_precise_loss2 = {}

for sigma in loss_precise:
    print(sigma)
    fit_sigma_temp1 = {}
    fit_sigma_temp2 = {}
    for epsilon in loss_precise[sigma]:
        processed_data = D_from_loss(loss_precise[sigma][epsilon][1:], sigma)
        fit_sigma_temp1[epsilon] = select_best_fit(
            non_linear_fit(
                dict(zip(n_turns, processed_data)),
                dict(zip(n_turns, processed_data * 0.01)),
                n_turns,
                method=1))
        fit_sigma_temp2[epsilon] = select_best_fit(
            non_linear_fit(
                dict(zip(n_turns, processed_data)),
                dict(zip(n_turns, processed_data * 0.01)),
                n_turns,
                method=2))
    fit_precise_loss1[sigma] = fit_sigma_temp1
    fit_precise_loss2[sigma] = fit_sigma_temp2

#%%

loss_precise_fit1 = {}
loss_precise_fit2 = {}

for sigma in sigmas:
    print(sigma)
    print("from fit1")

    loss_D_fit_temp = {}
    for epsilon in fit_precise_loss1[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit_precise_loss1[sigma][epsilon],
                                           FIT1, time, sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
    loss_precise_fit1[sigma] = loss_D_fit_temp

    print("from fit2")

    loss_D_fit_temp = {}
    for epsilon in fit_precise_loss2[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit_precise_loss2[sigma][epsilon],
                                           FIT2, time, sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
    loss_precise_fit2[sigma] = loss_D_fit_temp

#%%
print("Plot both loss fits.")

for sigma in sigmas:
    print(sigma)
    for epsilon in loss_D_fit1[sigma]:
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise[sigma][epsilon][1:],
            linewidth=0.5,
            label="Precise loss".format(N))
        for N in loss_D_fit1[sigma][epsilon]:
            #plt.plot(np.concatenate((np.array([0]),n_turns))[1:], loss_D_fit1[sigma][epsilon][N][1:], linewidth=0.5, label="D loss FIT1, N $= {}$".format(N))
            #plt.plot(np.concatenate((np.array([0]),n_turns))[1:], np.absolute(loss_precise[sigma][epsilon] - loss_D_fit1[sigma][epsilon][N])[1:], linewidth=0.5, label="Difference FIT1, N part $= {}$".format(N))
            plt.plot(
                np.concatenate((np.array([0]), n_turns))[1:],
                loss_D_fit2[sigma][epsilon][N][1:],
                linewidth=0.5,
                label="D loss FIT2, N $= {}$".format(N))
            #plt.plot(np.concatenate((np.array([0]),n_turns))[1:], np.absolute(loss_precise[sigma][epsilon] - loss_D_fit2[sigma][epsilon][N])[1:], linewidth=0.5, label="Difference FIT2, N part $= {}$".format(N))
        #plt.plot(np.concatenate((np.array([0]),n_turns))[1:], loss_precise_fit1[sigma][epsilon][1:], linewidth=0.5, label="D loss precise FIT1")
        #plt.plot(np.concatenate((np.array([0]),n_turns))[1:], loss_precise_fit2[sigma][epsilon][1:], linewidth=0.5, label="D loss precise FIT2")
        plt.xlabel("N turns")
        plt.xscale("log")
        plt.xlim(1e3, 1e7)
        plt.ylabel("Relative Intensity")
        plt.ylim(0, 1)
        plt.title(
            "Comparison of loss measures (FIT1 and FIT2), $\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
            format(sigma, epsilon[2]))
        plt.legend(prop={"size": 7})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "img/loss_both_sig{:2.2f}_eps{:2.0f}.png".format(
                sigma, epsilon[2]),
            dpi=DPI)
        plt.clf()

#%%
print("Compare loss fits with D fits (of 1 sector)")


def plot_4_different_fits(params1,
                          params2,
                          params3,
                          params4,
                          func1,
                          func2,
                          sigma,
                          epsilon,
                          n_turns=n_turns):
    plt.plot(
        n_turns,
        func1(n_turns, params1[0], params1[2], params1[4]),
        linewidth=0.5,
        label=
        'fit1 from D: $D_\infty={:4.2f}\pm{:4.2f}, B={:4.2f}\pm{:4.2f}, k={:4.2f}\pm{:4.2f}$'.
        format(params1[0], params1[1], params1[2], params1[3], params1[4],
               k_error))
    plt.plot(
        n_turns,
        func1(n_turns, params2[0], params2[2], params2[4]),
        linewidth=0.5,
        label=
        'fit1 Precise: $D_\infty={:4.2f}\pm{:4.2f}, B={:4.2f}\pm{:4.2f}, k={:4.2f}\pm{:4.2f}$'.
        format(params2[0], params2[1], params2[2], params2[3], params2[4],
               k_error))
    plt.plot(
        n_turns,
        func2(n_turns, params3[0], params3[2], params3[4]),
        linewidth=0.5,
        label=
        'fit2 from D: $A={:4.2f}\pm{:4.2f}, B={:4.2f}\pm{:4.2f}, k={:4.2f}\pm{:4.2f}$'.
        format(params3[0], params3[1], params3[2], params3[3], params3[4],
               k_error))
    plt.plot(
        n_turns,
        func2(n_turns, params4[0], params4[2], params4[4]),
        linewidth=0.5,
        label=
        'fit2 Precise: $A={:4.2f}\pm{:4.2f}, B={:4.2f}\pm{:4.2f}, k={:4.2f}\pm{:4.2f}$'.
        format(params4[0], params4[1], params4[2], params4[3], params4[4],
               k_error))
    plt.legend(prop={"size": 6})
    plt.grid(True)
    plt.xlabel("N Turns")
    plt.xscale("log")
    plt.ylabel("D (A.U.)")
    plt.title("Fit Comparison, $\sigma = {:2.2f}, \epsilon = {:2.2f}$.".format(
        sigma, epsilon[2]))
    plt.tight_layout()
    plt.savefig(
        "img/comparison_loss_precise_sigma{:2.2f}_eps{:2.2f}.png".format(
            sigma, epsilon[2]),
        dpi=DPI)
    plt.clf()


for sigma in sigmas:
    for epsilon in best_fit_parameters1:
        plot_4_different_fits(best_fit_parameters1[epsilon][1][np.pi / 4],
                              fit_precise_loss1[sigma][epsilon],
                              best_fit_parameters2[epsilon][1][np.pi / 4],
                              fit_precise_loss2[sigma][epsilon], FIT1, FIT2,
                              sigma, epsilon)


#%%
def remove_first_times_lhc(data, lower_bound):
    for folder in data:
        for kind in data[folder]:
            for seed in data[folder][kind]:
                for time in list(seed.keys()):
                    if time < lower_bound:
                        del seed[time]
    return data


print("LHC data.")
lhc_data = pickle.load(open("LHC_DATA.pkl", "rb"))

lhc_data = remove_first_times_lhc(lhc_data, 1000)

#%%
print("compute fits")


def sigma_filler(data_dict):
    sigma_dict = {}
    for element in data_dict:
        sigma_dict[element] = data_dict[element] * 0.01
    return sigma_dict


fit_lhc1 = {}
fit_lhc2 = {}
best_fit_lhc1 = {}
best_fit_lhc2 = {}

for label in lhc_data:
    fit_lhc1_label = {}
    fit_lhc2_label = {}
    best_fit_lhc1_label = {}
    best_fit_lhc2_label = {}
    for i in lhc_data[label]:
        print(label, i)
        fit_lhc1_correction = []
        fit_lhc2_correction = []
        best_fit_lhc1_correction = []
        best_fit_lhc2_correction = []
        for seed in lhc_data[label][i]:
            fit_lhc1_correction.append(
                non_linear_fit(
                    seed, sigma_filler(seed), list(seed.keys())[1:], method=1))
            best_fit_lhc1_correction.append(
                select_best_fit(fit_lhc1_correction[-1]))
            fit_lhc2_correction.append(
                non_linear_fit(
                    seed, sigma_filler(seed), list(seed.keys())[1:], method=2))
            best_fit_lhc2_correction.append(
                select_best_fit(fit_lhc2_correction[-1]))
        fit_lhc1_label[i] = fit_lhc1_correction
        fit_lhc2_label[i] = fit_lhc2_correction
        best_fit_lhc1_label[i] = best_fit_lhc1_correction
        best_fit_lhc2_label[i] = best_fit_lhc2_correction
    fit_lhc1[label] = fit_lhc1_label
    fit_lhc2[label] = fit_lhc2_label
    best_fit_lhc1[label] = best_fit_lhc1_label
    best_fit_lhc2[label] = best_fit_lhc2_label

# takes a long time, better to save the progress
fit_lhc = (fit_lhc1, fit_lhc2, best_fit_lhc1, best_fit_lhc2)
with open("LHC_FIT.pkl", "wb") as f:
    pickle.dump(fit_lhc, f, pickle.HIGHEST_PROTOCOL)

#%%
print("Load fit (if already done).")

fit_lhc = pickle.load(open("LHC_FIT.pkl", "rb"))
fit_lhc1 = fit_lhc[0]
fit_lhc2 = fit_lhc[1]
best_fit_lhc1 = fit_lhc[2]
best_fit_lhc2 = fit_lhc[3]

#%%
print("general lhc plots.")


def plot_lhc_fit(best_fit, data, func, label):
    for i in range(len(data)):
        plt.plot(
            sorted(data[i])[1:], [data[i][x] for x in sorted(data[i])[1:]],
            label="data {}".format(i),
            color="b",
            markersize=0.5,
            marker="x",
            linewidth=0)
        plt.plot(
            sorted(data[i])[1:],
            func(
                sorted(data[i])[1:], best_fit[i][0], best_fit[i][2],
                best_fit[i][4]),
            ('g--' if best_fit[i][2] > 0 and best_fit[i][4] > 0 else 'r--'),
            linewidth=0.5,
            label='fit {}'.format(i))
    plt.xlabel("N turns")
    #plt.xscale("log")
    plt.ylabel("D (A.U.)")
    plt.title("Tutto LHC, " + label)
    plt.tight_layout()
    plt.savefig("img/all_lhc_" + label + ".png", dpi=1000)
    plt.clf()


for folder in lhc_data:
    for kind in lhc_data[folder]:
        print(folder, kind)
        plot_lhc_fit(best_fit_lhc1[folder][kind], lhc_data[folder][kind], FIT1,
                     folder + kind + "F1")
        plot_lhc_fit(best_fit_lhc2[folder][kind], lhc_data[folder][kind], FIT2,
                     folder + kind + "F2")

#%%
print("lhc best fit distribution")


def best_fit_parameter_distribution(params,
                                    title="plot",
                                    namefile="plot",
                                    method=1):
    plt.errorbar(
        list(range(len(params))), [x[0] for x in params],
        yerr=[x[1] for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=1)
    plt.xlabel("Seed number")
    plt.ylabel(("A" if method == 2 else "$D_\infty$") + " parameter (A.U.)")
    plt.title(title + ", " + ("A" if method == 2 else "$D_\infty$") +
              " parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(
        "img/lhc" + namefile + "_" + ("A" if method == 2 else "Dinf") + ".png",
        dpi=DPI)
    plt.clf()

    plt.errorbar(
        list(range(len(params))), [x[2] for x in params],
        yerr=[x[3] for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=1)
    plt.xlabel("Seed number")
    plt.ylabel("B parameter (A.U.)")
    plt.title(title + ", B parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("img/lhc" + namefile + "_B.png", dpi=DPI)
    plt.clf()

    plt.errorbar(
        list(range(len(params))), [x[4] for x in params],
        yerr=[k_error for x in params],
        linewidth=0,
        elinewidth=1,
        marker="o",
        markersize=1)
    plt.xlabel("Seed number")
    plt.ylabel("k parameter (A.U.)")
    plt.title(title + ", k parameter")
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("img/lhc" + namefile + "_k.png", dpi=DPI)
    plt.clf()


for label in best_fit_lhc1:
    for i in best_fit_lhc1[label]:
        best_fit_parameter_distribution(best_fit_lhc1[label][i],
                                        label + ", FIT1, " + i,
                                        label + "_fit1_{}".format(i), 1)

for label in best_fit_lhc2:
    for i in best_fit_lhc2[label]:
        best_fit_parameter_distribution(best_fit_lhc2[label][i],
                                        label + ", FIT2, " + i,
                                        label + "_fit2_{}".format(i), 2)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
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
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.set_title(
        "Stable Region (angular scan)\n$(\omega_x = {:3.3f}, \omega_y = {:3.3f}, \epsilon = {:3.3f})$".
        format(key[0], key[1], key[2]))
    fig.tight_layout()
    fig.savefig(
        "img/stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(
            key[2], key[0], key[1]),
        dpi=DPI)
    plt.clf()

#%%
print("Draw 2D stability maps from linscan")

from matplotlib.colors import LogNorm

for epsilon in lin_data:
    temp = np.copy(lin_data[epsilon])
    temp += 1
    plt.imshow(
        temp,
        origin="lower",
        extent=(0, 0.8, 0, 0.8),
        norm=LogNorm(vmin=1, vmax=10000000))
    plt.xlabel("X coordinate (A.U.)")
    plt.ylabel("Y coordinate (A.U.)")
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.8)
    plt.grid(True)
    plt.title(
        "Stable Region (grid scan), number of turns\n$(\omega_x = {:3.3f}, \omega_y = {:3.3f}, \epsilon = {:3.3f})$".
        format(epsilon[0], epsilon[1], epsilon[2]))
    plt.colorbar()
    #plt.tight_layout()
    plt.savefig(
        "img/grid_stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(
            epsilon[2], epsilon[0], epsilon[1]),
        dpi=DPI)
    plt.clf()

#%%
# Concatenate
import cv2
for epsilon in lin_data:
    img1 = cv2.imread(
        "img/grid_stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(
            epsilon[2], epsilon[0], epsilon[1]))
    img2 = cv2.imread(
        "img/stability_eps{:2.0f}_wx{:3.3f}_wy{:3.3f}.png".format(
            epsilon[2], epsilon[0], epsilon[1]))
    vis = np.concatenate((img1, img2), axis=1)
    cv2.imwrite("img/concatenated_stability_eps{:2.0f}.png".format(epsilon[2]),
                vis)

#%%
# Compare every region

from matplotlib.colors import LogNorm

stability_levels = np.array(
    [10000000, 5000000, 1000000, 500000, 100000, 50000, 10000, 5000, 1000])

for epsilon in lin_data:
    for level in stability_levels:
        fig, ax = plt.subplots()
        temp = np.copy(lin_data[epsilon])
        temp += 1
        coso = ax.imshow(
            temp,
            origin="lower",
            extent=(0, 0.8, 0, 0.8),
            norm=LogNorm(vmin=level, vmax=10000000))
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
        ax.fill(x, y, "r", label="Angle scan".format(int(np.log10(level))))
        ax.legend()
        ax.set_xlabel("X coordinate (A.U.)")
        ax.set_ylabel("Y coordinate (A.U.)")
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.set_aspect("equal", "box")
        ax.grid(True)
        fig.colorbar(coso, ax=ax)
        ax.set_title(
            "Comparision of scans, $\epsilon = {:2.0f}$, $N = {}$".format(
                epsilon[2], level))
        fig.savefig(
            "img/comparison_eps{:2.0f}_N{}.png".format(epsilon[2], level),
            dpi=DPI)
        plt.close()

#%%
# Convert to JPEG

#converter.png_to_jpg("img/")
