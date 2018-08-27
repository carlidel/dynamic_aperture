import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os

#%%
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


def initialize(t_max, nt, I_abs, nI, dI, sigma):
    # t array
    t_array = np.linspace(0., t_max, num=nt)
    # I0 initialization
    I_array = np.linspace(dI, I_abs, num=nI)
    #intensity_array = I0_gaussian(I_array, sigma)
    intensity_array = I0_exponential(I_array, sigma)
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
        print(i)
        right = r_matrix.dot(sim_array[i - 1])
        sim_array[i] = np.linalg.solve(l_matrix, right)

    return t_array, I_array, sim_array


def relative_loss(I_array, sim_array):
    I0 = integrate.simps(sim_array[0], I_array)
    print("I0 = ", I0)
    intensity = []
    for line in sim_array:
        intensity.append(integrate.simps(line, I_array))
    intensity = np.asarray(intensity)
    return intensity / I0

#%%
# Parameters
epsilon = 10.0 * 1e-4

t_max = 50000000. * epsilon ** 2
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
print(intensity)

plt.clf()
plt.plot(t_array / (epsilon ** 2), intensity)

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
