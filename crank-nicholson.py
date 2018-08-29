import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
import os
import pickle

# Functions
def c_norm(I_abs, I_star, dI, nI, k):
    x_values = np.linspace(dI, I_abs, num=nI)
    y_values = np.exp(-2 * np.power((I_star / x_values), 1 / (2 * k)))
    return 1 / integrate.simps(y_values, x_values)


def D(I, I_star, k, c, I_abs):
    return c * np.exp(-2 * np.power(I_star / I, 1 / (2 * k)))


def I0_gaussian(I, sigma):
    return ((1 / (2 * np.pi * sigma * sigma)) * np.exp(-(I * I /
                                                         (2 * sigma * sigma))))


def I0_exponential(I, sigma):
    return 1 / (sigma * sigma) * np.exp(-I / (sigma * sigma))


def initialize(t_max, nt, I_abs, nI, dI, sigma, rho_0="exponential"):
    # t array
    t_array = np.linspace(0., t_max, num=nt)
    # I0 initialization
    I_array = np.linspace(dI, I_abs, num=nI)
    if rho_0 == "exponential":
        intensity_array = I0_exponential(I_array, sigma)
    elif rho_0 == "gaussian":
        intensity_array = I0_gaussian(I_array, sigma)
    else:
        assert False
    sim_array = np.zeros((nt, nI))
    sim_array[0] = intensity_array
    return t_array, I_array, intensity_array, sim_array


def initialize_l_matrix(nI, I_array, dI, dt, D, I_star, k, sigma, c, I_abs):
    l_matrix = np.zeros((nI, nI))
    for i in range(len(l_matrix)):
        if i == 0:
            l_matrix[i][0] = (
                D(I_array[0] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI) +
                D(I_array[0] - dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI) +
                1 / dt)
            l_matrix[i][1] = (
                -D(I_array[0] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI))
        elif i == len(l_matrix) - 1:
            l_matrix[i][-1] = (
                D(I_array[-1] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI)
                + D(I_array[-1] - dI * 0.5, I_star, k, c, I_abs) /
                (4 * dI * dI) + 1 / dt)
            l_matrix[i][-2] = (-D(I_array[-1] - dI * 0.5, I_star, k, c, I_abs)
                               / (4 * dI * dI))
        else:
            l_matrix[i][i - 1] = (
                -D(I_array[i] - dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI))
            l_matrix[i][i] = (
                D(I_array[i] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI) +
                D(I_array[i] - dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI) +
                1 / dt)
            l_matrix[i][i + 1] = (
                -D(I_array[i] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI))
    return l_matrix


def initialize_r_matrix(nI, I_array, dI, dt, D, I_star, k, sigma, c, I_abs):
    r_matrix = np.zeros((nI, nI))
    for i in range(len(r_matrix)):
        if i == 0:
            r_matrix[i][0] = (
                -D(I_array[0] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI)
                - D(I_array[0] - dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI)
                + 1 / dt)
            r_matrix[i][1] = (
                D(I_array[0] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI))
        elif i == len(r_matrix) - 1:
            r_matrix[i][-1] = (
                -D(I_array[-1] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI)
                - D(I_array[-1] - dI * 0.5, I_star, k, c, I_abs) /
                (4 * dI * dI) + 1 / dt)
            r_matrix[i][-2] = (
                D(I_array[-1] - dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI))
        else:
            r_matrix[i][i - 1] = (
                D(I_array[i] - dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI))
            r_matrix[i][i] = (
                -D(I_array[i] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI)
                - D(I_array[i] - dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI)
                + 1 / dt)
            r_matrix[i][i + 1] = (
                D(I_array[i] + dI * 0.5, I_star, k, c, I_abs) / (4 * dI * dI))
    return r_matrix


def execute_crank_nicolson(t_max, nt, dt, I_abs, nI, dI, sigma, I_star, k):
    c = c_norm(I_abs, I_star, dI, nI, k)
    t_array, I_array, intensity_array, sim_array = initialize(
        t_max, nt, I_abs, nI, dI, sigma)
    l_matrix = initialize_l_matrix(nI, I_array, dI, dt, D, I_star, k, sigma, c,
                                   I_abs)
    r_matrix = initialize_r_matrix(nI, I_array, dI, dt, D, I_star, k, sigma, c,
                                   I_abs)
    for i in range(1, len(sim_array)):
        #print(i)
        right = r_matrix.dot(sim_array[i - 1])
        sim_array[i] = np.linalg.solve(l_matrix, right)

    return t_array, I_array, sim_array


def relative_loss(I_array, sim_array):
    I0 = integrate.simps(sim_array[0], I_array)
    #print("I0 = ", I0)
    intensity = []
    for line in sim_array:
        intensity.append(integrate.simps(line, I_array))
    intensity = np.asarray(intensity)
    return intensity / I0


def find_point(function, target_value, min_point, max_point, precision=0.000000001):
    def g(x):
        return function(x) - target_value
    x0 = min_point
    x1 = max_point
    x2 = (x1 + x0) / 2
    while np.absolute(g(x2)) > precision:
        if g(x2) > 0.:
            x0 = x2
        elif g(x2) < 0.:
            x1 = x2
        else:
            return x2
        x2 = (x1 + x0) / 2
    return x2


def l2_computation(time_samples, data, epsilon, func):
    epsilon_2 = epsilon ** 2
    computed = np.array([func(x * epsilon_2) for x in time_samples])
    difference_squared = np.power(data - computed, 2)
    return integrate.simps(difference_squared, time_samples)


def diffusion_fitting(time_samples, data, I_abs, sigma, k, rho_0="exponential"):
    # I_abs è dato da considerazioni esterne
    nt = 1001
    dt = time_samples[-1] / nt
    nI = 100
    dI = I_abs / nI
    sigma = 1.
    I_star = 11.0
    dI_star = 1.
    dI_star_min = 0.001
    while dI_star >= dI_star_min:
        t_array, I_array, sim_array = execute_crank_nicolson(
            time_samples[-1], nt, dt, I_abs, nI, dI, sigma, I_star, k)
        t_array, I_array, sim_array_l = execute_crank_nicolson(
            time_samples[-1], nt, dt, I_abs, nI, dI, sigma, I_star - dI_star, k)
        t_array, I_array, sim_array_r = execute_crank_nicolson(
            time_samples[-1], nt, dt, I_abs, nI, dI, sigma, I_star + dI_star, k)
        intensity_array = relative_loss(I_array, sim_array)
        intensity_array_l = relative_loss(I_array, sim_array_l)
        intensity_array_r = relative_loss(I_array, sim_array_r)
        f = interpolate.interp1d(t_array, intensity_array, kind='cubic')
        f_l = interpolate.interp1d(t_array, intensity_array_l, kind='cubic')
        f_r = interpolate.interp1d(t_array, intensity_array_r, kind='cubic')
        point = find_point(f, data[-1], 0., time_samples[-1])
        point_l = find_point(f_l, data[-1], 0., time_samples[-1])
        point_r = find_point(f_r, data[-1], 0., time_samples[-1])
        epsilon = np.sqrt(point / time_samples[-1])
        epsilon_l = np.sqrt(point_l / time_samples[-1])
        epsilon_r = np.sqrt(point_r / time_samples[-1])
        print(point, time_samples[-1])
        print(point / epsilon**2)
        l2 = l2_computation(time_samples, data, epsilon, f)
        print(I_star, dI_star, l2)
        l2_l = l2_computation(time_samples, data, epsilon_l, f_l)
        l2_r = l2_computation(time_samples, data, epsilon_r, f_r)
        if l2 < l2_l and l2 < l2_r:
            dI_star /= 2.
        elif l2 > l2_l and l2 < l2_r:
            I_star -= dI_star
            epsilon = epsilon_l
        else:
            I_star += dI_star
            epsilon = epsilon_r
    return I_star, dI_star, epsilon

#%%
print("load data")

nek_data = pickle.load(open("data_nek_dictionary.pkl", "rb"))

time_samples = nek_data["6a"][0]
data = nek_data["6a"][1]

#%%
I_star, dI_star, epsilon = diffusion_fitting(time_samples, data, 7.8, 1., 0.33)

#%%
I_abs = 7.8
t_max = time_samples[-1]
nt = 1001
dt = t_max / nt
nI = 100
dI = I_abs / nI
sigma = 1.
k = 0.33

t_array, I_array, sim_array = execute_crank_nicolson(t_max, nt, dt, I_abs, nI,
                                                     dI, sigma, I_star, k)
intensity = relative_loss(I_array, sim_array)
f = interpolate.interp1d(t_array, intensity, kind='cubic')
#%%
plt.clf()
plt.plot(t_array, f(t_array * (epsilon**2)), label="crank")
plt.plot(time_samples, data, label="data")
#plt.xscale("log")
plt.savefig("roba.png", dpi=600)

#%%
# Parameters
epsilon = 10.0 * 1e-4

t_max = 50000000. * epsilon**2
nt = 10001
dt = t_max / nt

I_abs = 8.0
nI = 100
dI = I_abs / nI

sigma = 1.
I_star = 14.0
k = 0.33
#%%
t_array, I_array, sim_array = execute_crank_nicolson(t_max, nt, dt, I_abs, nI,
                                                     dI, sigma, I_star, k)

#%%
print("(trying) to compute relative loss")

intensity = relative_loss(I_array, sim_array)

plt.clf()
plt.plot(t_array / (epsilon**2), intensity, label="0")
plt.legend()

#%%
print("plot")

os.system("bash -c \"rm -f img*.png\"")
c = 0
for i in range(len(sim_array)):
    if i % 100 == 0:
        c += 1
        plt.plot(I_array, sim_array[i])
        #plt.ylim(0., 0.2)
        plt.savefig("img" + str(c).zfill(6) + ".png")
        plt.clf()
        print(c)
#%%
os.system("ffmpeg -y -i \"img%06d.png\" fok_cn.m4v")
#%%
os.system("bash -c \"rm -f img*.png\"")
