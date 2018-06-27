import numpy as np

# Parameters
dx = 1.e-4 # Delta space
T = 100000 # Number of turns

# Parameters of the modulated Hénnon Map
epsilon_k = [1.e-4, 0.218e-4, 0.708e-4, 0.254e-4, 0.1e-4, 0.078e-4, 0.218e-4]
Omega_k = [2 * np.pi / 868.12]
Omega_k.append(2 * Omega_k[0])
Omega_k.append(3 * Omega_k[0])
Omega_k.append(6 * Omega_k[0])
Omega_k.append(7 * Omega_k[0])
Omega_k.append(10 * Omega_k[0])
Omega_k.append(12 * Omega_k[0])

epsilon_k = np.asarray(epsilon_k)
Omega_k = np.asarray(Omega_k)

omega_x0 = 0.168
omega_y0 = 0.201
epsilon = 1

boundary = 2.0

def modulated_hennon_map(v0, n):
	'''
	v0 is the quadrimentional vector (x, px, y, py)
	n is the iteration number
	'''
	# Omegas computation
	omega_x = omega_x0 * (1 + epsilon * np.sum(epsilon_k * np.cos(Omega_k * n)))
	omega_y = omega_y0 * (1 + epsilon * np.sum(epsilon_k * np.cos(Omega_k * n)))
	# Linear matrix computation
	cosx = np.cos(omega_x)
	sinx = np.sin(omega_x)
	cosy = np.cos(omega_y)
	siny = np.sin(omega_y)
	L = np.array([[cosx, -sinx, 0, 0],[sinx, cosx, 0, 0],[0, 0, cosy, -siny],[0, 0, siny, cosy]])
	# Vector preparation
	v = np.array([v0[0], v0[1] + v0[0] * v0[0] - v0[2] * v0[2], v0[2], v0[3] - 2 * v0[0] * v0[2]])
	# Dot product
	return np.dot(L, v)

# Let's try with this thing...

def particle(x0, y0):
	#print("tracking particle ({},{})".format(x0,y0))
	v = np.array([x0, 0., y0, 0.])
	for i in range(T):
		v = modulated_hennon_map(v, i)
		if np.absolute(v[0]) > boundary or np.absolute(v[2]) > boundary:
			# particle lost!
			#print("particle ({},{}) lost at step {}.".format(x0,y0,i))
			return i
	# particle not lost!
	#print("particle ({},{}) survived.".format(x0,y0))
	return -1

# Single core (for now)
# Single line (for now)

values = [0, 1, 4, 16, 64]
times = [1000, 100000, 10000000]
survival_limits = []

for value in values:
	print(value)
	epsilon = value
	temp = []
	for time in times:
		print(time)
		T = time
		flag = True
		i = 0
		while flag:
			i += dx
			flag = (-1 == particle(i,0))
		temp.append(i)
	survival_limits.append(temp)
	print("Survival limit is {} for epsilon {}".format(i, epsilon))

survival_limits = np.asarray(survival_limits)
print(survival_limits)
np.save("survival_limits")