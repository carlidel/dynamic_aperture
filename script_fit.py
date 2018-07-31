from fit_library import *
import pickle
import numpy as np
import matplotlib.pyplot as plt

#%%
# Search parameters
k_max = 10.
k_min = -10.
dk = 0.1

A_max = 3.2
A_min = -100.
dA = 0.1

dx = 0.01

#%%
print("load data")

data = pickle.load(open("radscan_dx01_firstonly_dictionary.pkl", "rb"))
lin_data = pickle.load(open("linscan_dx01_firstonly_dictionary.pkl", "rb"))

# temporary removal of high epsilons for performance:
i = 0
for epsilon in sorted(data.keys()):
    if i > 4:
        del data[epsilon]
        del lin_data[epsilon]
    i += 1
## end temporary

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
            fit[angle] = non_linear_fit1(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns, k_min, k_max, dk)
            best[angle] = select_best_fit1(fit[angle])
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
            fit[angle] = non_linear_fit2(
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][0],
                dynamic_aperture[epsilon][len(partition_list) - 1][angle][1],
                n_turns, A_min, A_max, dA)
            best[angle] = select_best_fit2(fit[angle])
        fit_parameters_epsilon[len(partition_list) - 1] = fit
        best_fit_parameters_epsilon[len(partition_list) - 1] = best

    fit_parameters2[epsilon] = fit_parameters_epsilon
    best_fit_parameters2[epsilon] = best_fit_parameters_epsilon

#%%
print("Plot fits from simulation.")

for epsilon in best_fit_parameters1:
    #for n_angles in best_fit_parameters1[epsilon]:
    for angle in best_fit_parameters1[epsilon][1]:
        plot_fit_basic1(best_fit_parameters1[epsilon][1][angle],
                        1, epsilon, angle, n_turns,
                        dynamic_aperture)
        plot_fit_basic2(best_fit_parameters2[epsilon][1][angle],
                        1, epsilon, angle, n_turns,
                        dynamic_aperture)

#%%
print("Compare chi squared fits.")

for epsilon in fit_parameters1:
    for angle in fit_parameters1[epsilon][1]:
        plot_chi_squared1(fit_parameters1[epsilon][1][angle],
                                epsilon[2],
                                1,
                                angle)
        plot_chi_squared2(fit_parameters2[epsilon][1][angle],
                                epsilon[2],
                                1,
                                angle)


#%%

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

for sigma in sigmas:
    print(sigma)
    print("from fit1")

    loss_D_fit_temp = {}
    for epsilon in fit_precise_loss1[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(
                    fit_precise_loss1[sigma][epsilon],
                    pass_params_fit1, 
                    time, 
                    sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
    loss_precise_fit1[sigma] = loss_D_fit_temp

    print("from fit2")

    loss_D_fit_temp = {}
    for epsilon in fit_precise_loss2[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(
                    fit_precise_loss2[sigma][epsilon],
                    pass_params_fit2, 
                    time, 
                    sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
    loss_precise_fit2[sigma] = loss_D_fit_temp

#%%

loss_anglescan_fit1 = {}
loss_anglescan_fit2 = {}

for sigma in sigmas:
    print(sigma)
    print("from fit1")

    loss_D_fit_temp = {}
    for epsilon in fit_anglescan_loss1[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(
                    fit_anglescan_loss1[sigma][epsilon],
                    pass_params_fit1, 
                    time, 
                    sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
    loss_anglescan_fit1[sigma] = loss_D_fit_temp

    print("from fit2")

    loss_D_fit_temp = {}
    for epsilon in fit_anglescan_loss2[sigma]:
        print(epsilon)
        intensity_evolution = [1.]
        for time in n_turns:
            intensity_evolution.append(
                single_partition_intensity(
                    fit_anglescan_loss2[sigma][epsilon],
                    pass_params_fit2, 
                    time, 
                    sigma))
        loss_D_fit_temp[epsilon] = intensity_evolution
    loss_anglescan_fit2[sigma] = loss_D_fit_temp

#%%
print("Plot both loss fits.")

for sigma in sigmas:
    print(sigma)
    for epsilon in loss_precise[sigma]:
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise[sigma][epsilon][1:],
            linewidth=0.5,
            label="Precise loss".format(N))
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_anglescan[sigma][epsilon][1:],
            linewidth=0.5,
            label="Anglescan loss".format(N))
        # for N in loss_D_fit1[sigma][epsilon]:
        #     plt.plot(
        #         np.concatenate((np.array([0]), n_turns))[1:],
        #         loss_D_fit1[sigma][epsilon][N][1:],
        #         linewidth=0.5,
        #         label="D loss FIT1, N $= {}$".format(N))
        #     plt.plot(
        #         np.concatenate((np.array([0]), n_turns))[1:],
        #         np.absolute(loss_precise[sigma][epsilon] -
        #                     loss_D_fit1[sigma][epsilon][N])[1:],
        #         linewidth=0.5,
        #         label="Difference FIT1, N part $= {}$".format(N))
        #     plt.plot(
        #         np.concatenate((np.array([0]), n_turns))[1:],
        #         loss_D_fit2[sigma][epsilon][N][1:],
        #         linewidth=0.5,
        #         label="D loss FIT2, N $= {}$".format(N))
        #     plt.plot(
        #         np.concatenate((np.array([0]), n_turns))[1:],
        #         np.absolute(loss_precise[sigma][epsilon] -
        #                     loss_D_fit2[sigma][epsilon][N])[1:],
        #         linewidth=0.5,
        #         label="Difference FIT2, N part $= {}$".format(N))
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise_fit1[sigma][epsilon][1:],
            linewidth=0.5,
            label="D loss precise FIT1")
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_precise_fit2[sigma][epsilon][1:],
            linewidth=0.5,
            label="D loss precise FIT2")
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_anglescan_fit1[sigma][epsilon][1:],
            linewidth=0.5,
            label="D loss anglescan FIT1")
        plt.plot(
            np.concatenate((np.array([0]), n_turns))[1:],
            loss_anglescan_fit2[sigma][epsilon][1:],
            linewidth=0.5,
            label="D loss anglescan FIT2")
        plt.xlabel("N turns")
        plt.xscale("log")
        plt.xlim(1e3, 1e7)
        plt.ylabel("Relative Intensity")
        #plt.ylim(0, 1)
        plt.title(
            "Comparison of loss measures (FIT1 and FIT2), $\sigma = {:2.2f}$, $\epsilon = {:2.0f}$".
            format(sigma, epsilon[2]))
        plt.legend(prop={"size": 7})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "img/loss/loss_both_sig{:2.2f}_eps{:2.0f}.png".format(
                sigma, epsilon[2]),
            dpi=DPI)
        plt.clf()

#%%
print("LHC data.")
lhc_data = pickle.load(open("LHC_DATA.pkl", "rb"))
lhc_data = remove_first_times_lhc(lhc_data, 500)

#%%
print("compute fits")

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
        j = 0
        print(label, i)
        fit_lhc1_correction = []
        fit_lhc2_correction = []
        best_fit_lhc1_correction = []
        best_fit_lhc2_correction = []
        for seed in lhc_data[label][i]:
            print(j)
            j+=1
            fit_lhc1_correction.append(
                                non_linear_fit1(seed, 
                                                sigma_filler(seed, 0.05),
                                                list(sorted(seed.keys())),
                                                k_min,
                                                k_max,
                                                dk))
            best_fit_lhc1_correction.append(
                                select_best_fit1(fit_lhc1_correction[-1]))
            
            fit_lhc2_correction.append(
                                non_linear_fit2(seed, 
                                                sigma_filler(seed, 0.05),
                                                list(sorted(seed.keys())), 
                                                A_min, 
                                                A_max, 
                                                dA))
            best_fit_lhc2_correction.append(
                                select_best_fit2(fit_lhc2_correction[-1]))
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
