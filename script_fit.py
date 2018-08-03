import pickle
import numpy as np
import matplotlib.pyplot as plt

from fit_library import *

#%%
# Search parameters
k_max = 7.
k_min = -10.
dk = 0.1

dA = 0.0001
A_max = 0.01
A_min = 0.001 + dA ### under this value it is just wrong

dx = 0.01

#%%
print("load data")

data = pickle.load(open("radscan_dx01_firstonly_dictionary.pkl", "rb"))
lin_data = pickle.load(open("linscan_dx01_firstonly_dictionary.pkl", "rb"))

# temporary removal of high epsilons for performance:
#i = 0
#for epsilon in sorted(data.keys()):
#    if i > 9:
#        del data[epsilon]
#        del lin_data[epsilon]
#    i += 1
# end temporary

contour_data = {}
for epsilon in data:
    contour_data[epsilon] = make_countour_data(data[epsilon], n_turns, dx)

#%%
dynamic_aperture = {}

for epsilon in sorted(data):
    dyn_temp = {}
    for partition_list in partition_lists:
        dyn_temp[len(partition_list) - 1] = divide_and_compute(
            data[epsilon], n_turns, partition_list)
    dynamic_aperture[epsilon] = dyn_temp

#%%
print("Fit on Partitions1")

fit_parameters1 = {}
best_fit_parameters1 = {}

for epsilon in dynamic_aperture:
    print(epsilon)
    # fit1
    fit_parameters_epsilon = {}
    best_fit_parameters_epsilon = {}

    for partition_list in partition_lists:
        fit = {}
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list) - 1]:
            fit[angle] = non_linear_fit1(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns, k_min, k_max, dk)
            best[angle] = select_best_fit1(fit[angle])
        fit_parameters_epsilon[len(partition_list) - 1] = fit
        best_fit_parameters_epsilon[len(partition_list) - 1] = best

    fit_parameters1[epsilon] = fit_parameters_epsilon
    best_fit_parameters1[epsilon] = best_fit_parameters_epsilon

#%%
print("Fit on Partitions2")

dA = 0.0001
A_max = 0.01
A_min = 0.001 ### under this value it doesn't converge

fit_parameters2 = {}
best_fit_parameters2 = {}

for epsilon in dynamic_aperture:
    print(epsilon)
    # fit2
    fit_parameters_epsilon = {}
    best_fit_parameters_epsilon = {}

    for partition_list in partition_lists:
        fit = {}
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list) - 1]:
            scale_search = 1.
            print(scale_search)
            fit[angle] = non_linear_fit2(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns,
                A_min,
                A_max * scale_search,
                dA * scale_search)
            best[angle] = select_best_fit2(fit[angle])
            ### Is this a naive minimum in the chi squared?
            while (best[angle][4] >= A_max * scale_search - dA * scale_search
                   and scale_search <= 1e50):
                print("Minimum naive! Increase scale_search!")
                A_min_new = A_max * scale_search
                scale_search *= 10.
                if scale_search > 1e50:
                    print("Maximum scale reached! This will be the last fit.")
                print(scale_search)
                fit[angle] = non_linear_fit2(
                    dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                    dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                    n_turns,
                    A_min_new,
                    A_max * scale_search,
                    dA * scale_search)
                best[angle] = select_best_fit2(fit[angle])
        fit_parameters_epsilon[len(partition_list) - 1] = fit
        best_fit_parameters_epsilon[len(partition_list) - 1] = best

    fit_parameters2[epsilon] = fit_parameters_epsilon
    best_fit_parameters2[epsilon] = best_fit_parameters_epsilon

#%%

dA = 0.001
A_max = 5.
A_min = 0.00
shift = - np.log10(n_turns[0]) + dA
    

print("Fit on Partitions2v1")

fit_parameters2_v1 = {}
best_fit_parameters2_v1 = {}

for epsilon in dynamic_aperture:
    print(epsilon)
    # fit2
    fit_parameters_epsilon = {}
    best_fit_parameters_epsilon = {}

    for partition_list in partition_lists:
        fit = {}
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list) - 1]:
            scale_search = 1.
            print(scale_search)
            fit[angle] = non_linear_fit2_v1(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns,
                A_min + shift,
                A_max + shift,
                dA)
            best[angle] = select_best_fit2_v1(fit[angle])
            ### Is this a naive minimum in the chi squared?
            while (best[angle][4] >= A_max * scale_search - dA * scale_search +
                   shift and scale_search <= 1e50):
                print("Minimum naive! Increase scale_search!")
                A_min_new = A_max * scale_search
                scale_search *= 10.
                if scale_search > 1e50:
                    print("Maximum scale reached! This will be the last fit.")
                print(scale_search)
                fit[angle] = non_linear_fit2_v1(
                    dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                    dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                    n_turns,
                    A_min + shift,
                    A_max * scale_search + shift,
                    dA * scale_search)
                best[angle] = select_best_fit2_v1(fit[angle])
        fit_parameters_epsilon[len(partition_list) - 1] = fit
        best_fit_parameters_epsilon[len(partition_list) - 1] = best

    fit_parameters2_v1[epsilon] = fit_parameters_epsilon
    best_fit_parameters2_v1[epsilon] = best_fit_parameters_epsilon

#%%
print("Fit naive on simulation 2. (in order to test it!)")

best_fit_parameters2 = {}
for epsilon in dynamic_aperture:
    print(epsilon)
    # fit2
    best_fit_parameters_epsilon = {}

    for partition_list in partition_lists:
        best = {}
        for angle in dynamic_aperture[epsilon][len(partition_list) - 1]:
            best[angle] = non_linear_fit2_naive(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns,
                0.1, 10., 10.1)
        best_fit_parameters_epsilon[len(partition_list) - 1] = best
    best_fit_parameters2[epsilon] = best_fit_parameters_epsilon

#%%
print("Plot fits from simulation 1.")
for epsilon in best_fit_parameters1:
    #for n_angles in best_fit_parameters1[epsilon]:
    for angle in best_fit_parameters1[epsilon][1]:
        plot_fit_basic1(best_fit_parameters1[epsilon][1][angle],
                        1, epsilon, angle, n_turns,
                        dynamic_aperture)
#%%
print("Plot fits from simulation 2.")
for epsilon in best_fit_parameters2:
    #for n_angles in best_fit_parameters1[epsilon]:
    for angle in best_fit_parameters2[epsilon][1]:
        plot_fit_basic2(best_fit_parameters2[epsilon][1][angle],
                        1, epsilon, angle, n_turns,
                        dynamic_aperture)

#%%
print("Plot fits from simulation 2 v1.")
for epsilon in best_fit_parameters2_v1:
    #for n_angles in best_fit_parameters1[epsilon]:
    for angle in best_fit_parameters2_v1[epsilon][1]:
        plot_fit_basic2_v1(best_fit_parameters2_v1[epsilon][1][angle],
                           1, epsilon, angle, n_turns,
                           dynamic_aperture)

#%%
print("Compare chi squared fits1.")
for epsilon in fit_parameters1:
    for angle in fit_parameters1[epsilon][1]:
        plot_chi_squared1(fit_parameters1[epsilon][1][angle],
                          epsilon[2],
                          1,
                          angle)

#%%
print("Compare chi squared fits2.")
for epsilon in fit_parameters2:
    for angle in fit_parameters2[epsilon][1]:
        plot_chi_squared2(fit_parameters2[epsilon][1][angle],
                          epsilon[2],
                          1,
                          angle)

#%%
print("Compare chi squared fits2 v1.")
for epsilon in fit_parameters2_v1:
    for angle in fit_parameters2_v1[epsilon][1]:
        plot_chi_squared2_v1(fit_parameters2_v1[epsilon][1][angle],
                          epsilon[2],
                          1,
                          angle)

#%%
print("Is this loss?")

# Weights at beginning

sigmas = [0.2, 0.25, 0.5, 0.75, 1]

loss_precise = {}
loss_anglescan = {}
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

    print("anglescan")

    loss_anglescan_temp = {}
    for epsilon in lin_data:
        print(epsilon)
        intensity_evolution = [1]
        for time in n_turns:
            intensity_evolution.append(
                loss_from_anglescan(contour_data[epsilon], time, sigma))
        loss_anglescan_temp[epsilon] = np.asarray(intensity_evolution)
    loss_anglescan[sigma] = loss_anglescan_temp

    print("from fit1")

    loss_D_fit_temp = {}
    loss_D_fit_temp_err = {}
    for epsilon in best_fit_parameters1:
        print(epsilon)
        loss_D_fit_temp_part = {}
        loss_D_fit_temp_part_err = {}
        for N in best_fit_parameters1[epsilon]:
            intensity_evolution = [1.]
            error_evolution = [0.]
            for time in n_turns:
                intensity_evolution.append(
                    multiple_partition_intensity(
                                            best_fit_parameters1[epsilon][N],
                                            pass_params_fit1,
                                            N,
                                            time,
                                            sigma))
                error_evolution.append(
                    error_loss_estimation(best_fit_parameters1[epsilon][N],
                                          pass_params_fit1,
                                          contour_data[epsilon],
                                          N,
                                          time,
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
            error_evolution = [0.]
            for time in n_turns:
                intensity_evolution.append(
                    multiple_partition_intensity(
                                            best_fit_parameters2[epsilon][N],
                                            pass_params_fit2,
                                            N,
                                            time,
                                            sigma))
                error_evolution.append(
                    error_loss_estimation(best_fit_parameters2[epsilon][N],
                                          pass_params_fit2,
                                          contour_data[epsilon],
                                          N,
                                          time,
                                          sigma))
            loss_D_fit_temp_part[N] = np.asarray(intensity_evolution)
            loss_D_fit_temp_part_err[N] = np.asarray(error_evolution)
            #print(loss_D_fit_temp_part[N])
        loss_D_fit_temp[epsilon] = loss_D_fit_temp_part
        loss_D_fit_temp_err[epsilon] = loss_D_fit_temp_part_err
    loss_D_fit2[sigma] = loss_D_fit_temp
    loss_D_fit2_err[sigma] = loss_D_fit_temp_err

#%%
print("Reverse engeneering dynamic_aperture")

processed_data_precise = {}
processed_data_anglescan = {}

for sigma in loss_precise:
    processed_data_precise_temp = {}
    processed_data_anglescan_temp = {}
    for epsilon in loss_precise[sigma]:
        processed_data_precise_temp[epsilon] = D_from_loss(
            np.copy(loss_precise[sigma][epsilon][1:]),
            sigma)
        processed_data_anglescan_temp[epsilon] = D_from_loss(
            np.copy(loss_anglescan[sigma][epsilon][1:]),
            sigma)
    processed_data_precise[sigma] = processed_data_precise_temp
    processed_data_anglescan[sigma] = processed_data_anglescan_temp

#%%

print("Fit on precise loss.")

fit_precise_loss1 = {}
fit_precise_loss2 = {}

for sigma in loss_precise:
    print(sigma)
    fit_sigma_temp1 = {}
    fit_sigma_temp2 = {}
    for epsilon in loss_precise[sigma]:
        fit_sigma_temp1[epsilon] = select_best_fit1(
            non_linear_fit1(
                dict(zip(n_turns, processed_data_precise[sigma][epsilon])),
                dict(zip(n_turns, processed_data_precise[sigma][epsilon] * 0.01)),
                n_turns,
                k_min,
                k_max,
                dk))
        fit_sigma_temp2[epsilon] = select_best_fit2(
            non_linear_fit2(
                dict(zip(n_turns, processed_data_precise[sigma][epsilon])),
                dict(zip(n_turns, processed_data_precise[sigma][epsilon] * 0.01)),
                n_turns,
                A_min,
                A_max,
                dA))
    fit_precise_loss1[sigma] = fit_sigma_temp1
    fit_precise_loss2[sigma] = fit_sigma_temp2

#%%
print("Fit on anglescan loss")

fit_anglescan_loss1 = {}
fit_anglescan_loss2 = {}

for sigma in loss_anglescan:
    print(sigma)
    fit_sigma_temp1 = {}
    fit_sigma_temp2 = {}
    for epsilon in loss_anglescan[sigma]:
        processed_data = D_from_loss(
            np.copy(loss_anglescan[sigma][epsilon][1:]),
            sigma)
        fit_sigma_temp1[epsilon] = select_best_fit1(
            non_linear_fit1(
                dict(zip(n_turns, processed_data_anglescan[sigma][epsilon])),
                dict(zip(n_turns, processed_data_anglescan[sigma][epsilon] * 0.01)),
                n_turns,
                k_min,
                k_max,
                dk))
        fit_sigma_temp2[epsilon] = select_best_fit2(
            non_linear_fit2(
                dict(zip(n_turns, processed_data_anglescan[sigma][epsilon])),
                dict(zip(n_turns, processed_data_anglescan[sigma][epsilon] * 0.01)),
                n_turns,
                A_min,
                A_max,
                dA))
    fit_anglescan_loss1[sigma] = fit_sigma_temp1
    fit_anglescan_loss2[sigma] = fit_sigma_temp2

#%%

loss_precise_fit1 = {}
loss_precise_fit2 = {}
loss_precise_fit1_err = {}
loss_precise_fit2_err = {}

for sigma in sigmas:
    print(sigma)
    print("from fit1")

    loss_D_fit_temp = {}
    loss_D_fit_temp_err = {}
    for epsilon in fit_precise_loss1[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        error_evolution = [0.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit_precise_loss1[sigma][epsilon],
                                           pass_params_fit1,
                                           time,
                                           sigma))
            error_evolution.append(
                error_loss_estimation_single_partition(
                                            fit_precise_loss1[sigma][epsilon],
                                            pass_params_fit1,
                                            contour_data[epsilon],
                                            time,
                                            sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
        loss_D_fit_temp_err[epsilon] = error_evolution
    loss_precise_fit1[sigma] = loss_D_fit_temp
    loss_precise_fit1_err[sigma] = loss_D_fit_temp_err

    print("from fit2")

    loss_D_fit_temp = {}
    loss_D_fit_temp_err = {}
    for epsilon in fit_precise_loss2[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        error_evolution = [0.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit_precise_loss2[sigma][epsilon],
                                           pass_params_fit2,
                                           time,
                                           sigma))
            error_evolution.append(
                error_loss_estimation_single_partition(
                                            fit_precise_loss1[sigma][epsilon],
                                            pass_params_fit2,
                                            contour_data[epsilon],
                                            time,
                                            sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
        loss_D_fit_temp_err[epsilon] = error_evolution
    loss_precise_fit2[sigma] = loss_D_fit_temp
    loss_precise_fit2_err[sigma] = loss_D_fit_temp_err

#%%

loss_anglescan_fit1 = {}
loss_anglescan_fit2 = {}
loss_anglescan_fit1_err = {}
loss_anglescan_fit2_err = {}

for sigma in sigmas:
    print(sigma)
    print("from fit1")

    loss_D_fit_temp = {}
    loss_D_fit_temp_err = {}
    for epsilon in fit_anglescan_loss1[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        error_evolution = [0.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit_anglescan_loss1[sigma][epsilon],
                                           pass_params_fit1,
                                           time,
                                           sigma))
            error_evolution.append(
                error_loss_estimation_single_partition(
                                            fit_anglescan_loss1[sigma][epsilon],
                                            pass_params_fit1,
                                            contour_data[epsilon],
                                            time,
                                            sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
        loss_D_fit_temp_err[epsilon] = error_evolution
    loss_anglescan_fit1[sigma] = loss_D_fit_temp
    loss_anglescan_fit1_err[sigma] = loss_D_fit_temp_err

    print("from fit2")

    loss_D_fit_temp = {}
    loss_D_fit_temp_err = {}
    for epsilon in fit_anglescan_loss2[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        error_evolution = [0.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(fit_anglescan_loss2[sigma][epsilon],
                                           pass_params_fit2,
                                           time,
                                           sigma))
            error_evolution.append(
                error_loss_estimation_single_partition(
                                            fit_anglescan_loss2[sigma][epsilon],
                                            pass_params_fit2,
                                            contour_data[epsilon],
                                            time,
                                            sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
        loss_D_fit_temp_err[epsilon] = error_evolution
    loss_anglescan_fit2[sigma] = loss_D_fit_temp
    loss_anglescan_fit2_err[sigma] = loss_D_fit_temp_err

#%%
print("Plot the losses!")

for sigma in sigmas:
    print(sigma)
    for epsilon in loss_precise[sigma]:
        print(epsilon)
        ### Just the losses (no fits)
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise[sigma][epsilon][1:],
            linewidth=0.5,
            label="Precise loss")
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_anglescan[sigma][epsilon][1:],
            linewidth=0.5,
            label="Anglescan loss")
        plt.xlabel("N turns")
        plt.xscale("log")
        plt.xlim(1e3, 1e7)
        plt.ylabel("Relative Intensity")
        #plt.ylim(0, 1)
        plt.title(
            "Comparison of loss measures (PRECISE and ANGLESCAN),\n" +
            "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".format(sigma, epsilon[2]))
        plt.legend(prop={"size": 7})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "img/loss/loss_precise_anglescan_sig{:2.2f}_eps{:2.0f}.png".format(
                sigma, epsilon[2]), dpi=DPI)
        plt.clf()

        ### Precise and Precise Fit
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise[sigma][epsilon][1:],
            linewidth=0.5,
            label="Precise loss")
        plt.errorbar(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise_fit1[sigma][epsilon][1:],
            yerr=loss_precise_fit1_err[sigma][epsilon][1:],
            linewidth=0.5,
            label="D loss precise FIT1")
        plt.errorbar(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise_fit2[sigma][epsilon][1:],
            yerr=loss_precise_fit2_err[sigma][epsilon][1:],
            linewidth=0.5,
            label="D loss precise FIT2")
        plt.title(
            "Comparison of loss measures (PRECISE with PRECISE FITS),\n" +
            "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".format(sigma, epsilon[2]))
        plt.xlabel("N turns")
        plt.xscale("log")
        plt.xlim(1e3, 1e7)
        plt.ylabel("Relative Intensity")
        plt.legend(prop={"size": 7})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "img/loss/loss_precise_and_fits_sig{:2.2f}_eps{:2.0f}.png".format(
                sigma, epsilon[2]), dpi=DPI)
        plt.clf()

        ### Anglescan and Anglescan Fit
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_anglescan[sigma][epsilon][1:],
            linewidth=0.5,
            label="Anglescan loss")
        plt.errorbar(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_anglescan_fit1[sigma][epsilon][1:],
            yerr=loss_anglescan_fit1_err[sigma][epsilon][1:],
            linewidth=0.5,
            label="D loss anglescan FIT1")
        plt.errorbar(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_anglescan_fit2[sigma][epsilon][1:],
            yerr=loss_anglescan_fit2_err[sigma][epsilon][1:],
            linewidth=0.5,
            label="D loss anglescan FIT2")
        plt.xlabel("N turns")
        plt.xscale("log")
        plt.xlim(1e3, 1e7)
        plt.ylabel("Relative Intensity")
        #plt.ylim(0, 1)
        plt.title(
            "Comparison of loss measures (ANGLESCAN with ANGLESCAN FITS),\n" +
            "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".format(sigma, epsilon[2]))
        plt.legend(prop={"size": 7})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "img/loss/loss_anglescan_and_fits_sig{:2.2f}_eps{:2.0f}.png".format(
                sigma, epsilon[2]), dpi=DPI)
        plt.clf()

        ### Precise and D Fits
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise[sigma][epsilon][1:],
            linewidth=0.5,
            label="Precise loss")
        for N in loss_D_fit1[sigma][epsilon]:
            plt.errorbar(
                np.concatenate((np.array([0]), n_turns))[1:],
                loss_D_fit1[sigma][epsilon][N][1:],
                yerr = loss_D_fit1_err[sigma][epsilon][N][1:],
                linewidth=0.5,
                label="D loss FIT1, N $= {}$".format(N))
            # plt.plot(
            #     np.concatenate((np.array([0]), n_turns))[1:],
            #     np.absolute(loss_precise[sigma][epsilon] -
            #                 loss_D_fit1[sigma][epsilon][N])[1:],
            #     linewidth=0.5,
            #     label="Difference FIT1, N part $= {}$".format(N))
            plt.errorbar(
                np.concatenate((np.array([0]), n_turns))[1:],
                loss_D_fit2[sigma][epsilon][N][1:],
                yerr = loss_D_fit2_err[sigma][epsilon][N][1:],
                linewidth=0.5,
                label="D loss FIT2, N $= {}$".format(N))
            # plt.plot(
            #     np.concatenate((np.array([0]), n_turns))[1:],
            #     np.absolute(loss_precise[sigma][epsilon] -
            #                 loss_D_fit2[sigma][epsilon][N])[1:],
            #     linewidth=0.5,
            #     label="Difference FIT2, N part $= {}$".format(N))
        plt.xlabel("N turns")
        plt.xscale("log")
        plt.xlim(1e3, 1e7)
        plt.ylabel("Relative Intensity")
        #plt.ylim(0, 1)
        plt.title(
            "Comparison of loss measures (PRECISE with D FITS),\n" +
            "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".format(sigma, epsilon[2]))
        plt.legend(prop={"size": 7})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "img/loss/loss_precise_and_Dfits_sig{:2.2f}_eps{:2.0f}.png".format(
                sigma, epsilon[2]), dpi=DPI)
        plt.clf()

        ### Anglescan and D Fits
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_anglescan[sigma][epsilon][1:],
            linewidth=0.5,
            label="Anglescan loss")
        for N in loss_D_fit1[sigma][epsilon]:
            plt.errorbar(
                np.concatenate((np.array([0]), n_turns))[1:],
                loss_D_fit1[sigma][epsilon][N][1:],
                yerr = loss_D_fit1_err[sigma][epsilon][N][1:],
                linewidth=0.5,
                label="D loss FIT1, N $= {}$".format(N))
            # plt.plot(
            #     np.concatenate((np.array([0]), n_turns))[1:],
            #     np.absolute(loss_precise[sigma][epsilon] -
            #                 loss_D_fit1[sigma][epsilon][N])[1:],
            #     linewidth=0.5,
            #     label="Difference FIT1, N part $= {}$".format(N))
            plt.errorbar(
                np.concatenate((np.array([0]), n_turns))[1:],
                loss_D_fit2[sigma][epsilon][N][1:],
                yerr = loss_D_fit2_err[sigma][epsilon][N][1:],
                linewidth=0.5,
                label="D loss FIT2, N $= {}$".format(N))
            # plt.plot(
            #     np.concatenate((np.array([0]), n_turns))[1:],
            #     np.absolute(loss_precise[sigma][epsilon] -
            #                 loss_D_fit2[sigma][epsilon][N])[1:],
            #     linewidth=0.5,
            #     label="Difference FIT2, N part $= {}$".format(N))
        plt.xlabel("N turns")
        plt.xscale("log")
        plt.xlim(1e3, 1e7)
        plt.ylabel("Relative Intensity")
        #plt.ylim(0, 1)
        plt.title(
            "Comparison of loss measures (ANGLESCAN with D FITS),\n" +
            "$\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".format(sigma, epsilon[2]))
        plt.legend(prop={"size": 7})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "img/loss/loss_anglescan_and_Dfits_sig{:2.2f}_eps{:2.0f}.png".format(
                sigma, epsilon[2]), dpi=DPI)
        plt.clf()
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt

from fit_library import *

#%%
# Search parameters
k_max = 7.
k_min = -10.
dk = 0.1

dA = 0.0001
A_max = 0.01
A_min = 0.001 ### under this value it doesn't converge

#%%
print("LHC data.")
lhc_data = pickle.load(open("LHC_DATA.pkl", "rb"))
lhc_data = remove_first_times_lhc(lhc_data, 1000)

#%%
print("compute fits1")

fit_lhc1 = {}
best_fit_lhc1 = {}

for label in lhc_data:
    fit_lhc1_label = {}
    best_fit_lhc1_label = {}
    for i in lhc_data[label]:
        j = 0
        print(label, i)
        fit_lhc1_correction = []
        best_fit_lhc1_correction = []
        for seed in lhc_data[label][i]:
            print(j)
            j += 1
            # FIT1
            fit_lhc1_correction.append(
                                non_linear_fit1(seed,
                                                sigma_filler(seed, 0.05),
                                                np.asarray(sorted(seed.keys())),
                                                k_min,
                                                k_max,
                                                dk))
            best_fit_lhc1_correction.append(
                                select_best_fit1(fit_lhc1_correction[-1]))

        fit_lhc1_label[i] = fit_lhc1_correction
        best_fit_lhc1_label[i] = best_fit_lhc1_correction
    fit_lhc1[label] = fit_lhc1_label
    best_fit_lhc1[label] = best_fit_lhc1_label

#%%
print("compute fits2")

fit_lhc2 = {}
best_fit_lhc2 = {}

for label in lhc_data:
    fit_lhc2_label = {}
    best_fit_lhc2_label = {}
    for i in lhc_data[label]:
        j = 0
        print(label, i)
        fit_lhc2_correction = []
        best_fit_lhc2_correction = []
        for seed in lhc_data[label][i]:
            print(j)
            j += 1
            scale_search = 1.
            # FIT2
            fit_lhc2_correction.append(
                                non_linear_fit2(seed,
                                                sigma_filler(seed, 0.05),
                                                np.asarray(sorted(seed.keys())),
                                                A_min,
                                                A_max,
                                                dA))
            best_fit_lhc2_correction.append(
                                select_best_fit2(fit_lhc2_correction[-1]))
            ### Is this a naive minimum in the chi squared?
            while (best_fit_lhc2_correction[-1][4] >= (A_max * scale_search 
                    - dA * scale_search) and scale_search <= 1e20):
                print("Minimum naive! Increase scale_search!")
                A_min_new = A_max * scale_search
                scale_search *= 10.
                if scale_search > 1e20:
                    print("Maximum scale reached! This will be the last fit.")
                print(scale_search)
                fit_lhc2_correction[-1] = non_linear_fit2(
                                                seed,
                                                sigma_filler(seed, 0.05),
                                                np.asarray(sorted(seed.keys())),
                                                A_min_new,
                                                A_max * scale_search,
                                                dA * scale_search)
                best_fit_lhc2_correction[-1] = select_best_fit2(
                                                fit_lhc2_correction[-1])
        fit_lhc2_label[i] = fit_lhc2_correction
        best_fit_lhc2_label[i] = best_fit_lhc2_correction
    fit_lhc2[label] = fit_lhc2_label
    best_fit_lhc2[label] = best_fit_lhc2_label

#%%
print("Compute FIT2 v1")

dA = 0.01
A_max = 2.
A_min = 0.00
shift = - np.log10(1000) + dA

fit_lhc2 = {}
best_fit_lhc2 = {}

for label in lhc_data:
    fit_lhc2_label = {}
    best_fit_lhc2_label = {}
    for i in lhc_data[label]:
        j = 0
        print(label, i)
        fit_lhc2_correction = []
        best_fit_lhc2_correction = []
        for seed in lhc_data[label][i]:
            print(j)
            j += 1
            scale_search = 1.
            # FIT2
            fit_lhc2_correction.append(
                                non_linear_fit2_v1(seed,
                                                sigma_filler(seed, 0.05),
                                                np.asarray(sorted(seed.keys())),
                                                A_min + shift,
                                                A_max + shift,
                                                dA))
            best_fit_lhc2_correction.append(
                                select_best_fit2(fit_lhc2_correction[-1]))
            ### Is this a naive minimum in the chi squared?
            while (best_fit_lhc2_correction[-1][4] >= (A_max * scale_search 
                    - dA * scale_search + shift) and scale_search <= 1e20):
                print("Minimum naive! Increase scale_search!")
                A_min_new = A_max * scale_search
                scale_search *= 10.
                if scale_search > 1e20:
                    print("Maximum scale reached! This will be the last fit.")
                print(scale_search)
                fit_lhc2_correction[-1] = non_linear_fit2_v1(
                                                seed,
                                                sigma_filler(seed, 0.05),
                                                np.asarray(sorted(seed.keys())),
                                                A_min + shift,
                                                A_max * scale_search + shift,
                                                dA * scale_search)
                best_fit_lhc2_correction[-1] = select_best_fit2_v1(
                                                fit_lhc2_correction[-1])
        fit_lhc2_label[i] = fit_lhc2_correction
        best_fit_lhc2_label[i] = best_fit_lhc2_correction
    fit_lhc2[label] = fit_lhc2_label
    best_fit_lhc2[label] = best_fit_lhc2_label

#%%
print("Compute (stupid) FIT2")
best_fit_lhc2 = {}

for label in lhc_data:
    best_fit_lhc2_label = {}
    for i in lhc_data[label]:
        j = 0
        print(label, i)
        best_fit_lhc2_correction = []
        for seed in lhc_data[label][i]:
            print(j)
            j += 1
            # FIT2
            best_fit_lhc2_correction.append(
                                non_linear_fit2_naive(seed,
                                                sigma_filler(seed, 0.05),
                                                np.asarray(sorted(seed.keys())),
                                                0.1, 0.1, 1))
        best_fit_lhc2_label[i] = best_fit_lhc2_correction
    best_fit_lhc2[label] = best_fit_lhc2_label

#%%
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

for folder in lhc_data:
    for kind in lhc_data[folder]:
        print(folder, kind)
        plot_lhc_fit(best_fit_lhc1[folder][kind], lhc_data[folder][kind],
                     pass_params_fit1, folder + kind + "f1")
        plot_lhc_fit(best_fit_lhc2[folder][kind], lhc_data[folder][kind],
                     pass_params_fit2, folder + kind + "f2")

#%%
print("lhc best fit distribution")

for label in best_fit_lhc1:
    for kind in best_fit_lhc1[label]:
        best_fit_seed_distrib1(best_fit_lhc1[label][kind], label + kind + "f1")
        lhc_2param_comparison1(best_fit_lhc1[label][kind], label + kind + "f1")
        lhc_plot_chi_squared1(fit_lhc1[label][kind], label, kind)
        combine_plots_lhc1(label, kind)
#%%

for label in best_fit_lhc2:
    for kind in best_fit_lhc2[label]:
        best_fit_seed_distrib2(best_fit_lhc2[label][kind], label + kind + "f2")
        lhc_2param_comparison2(best_fit_lhc2[label][kind], label + kind + "f2")
        lhc_plot_chi_squared2(fit_lhc2[label][kind], label, kind)
        combine_plots_lhc2(label, kind)
