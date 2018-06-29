import numpy as np

# Parameters of the modulated Hénnon Map
epsilon_k = [1.000e-4,
			 0.218e-4, 
			 0.708e-4, 
			 0.254e-4, 
			 0.100e-4, 
			 0.078e-4, 
			 0.218e-4]

Omega_k = [2 * np.pi / 868.12]
Omega_k.append(2  * Omega_k[0])
Omega_k.append(3  * Omega_k[0])
Omega_k.append(6  * Omega_k[0])
Omega_k.append(7  * Omega_k[0])
Omega_k.append(10 * Omega_k[0])
Omega_k.append(12 * Omega_k[0])

epsilon_k = np.asarray(epsilon_k)
Omega_k = np.asarray(Omega_k)

omega_x0 = 0.168 * 2 * np.pi
omega_y0 = 0.201 * 2 * np.pi

def modulated_henon_map(v0, epsilon, n):
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
	L = np.array([[cosx, sinx, 0, 0],
				  [-sinx, cosx, 0, 0],
				  [0, 0, cosy, siny],
				  [0, 0, -siny, cosy]])
	# Vector preparation
	v = np.array([v0[0],
				  v0[1] + v0[0] * v0[0] - v0[2] * v0[2],
				  v0[2],
				  v0[3] - 2 * v0[0] * v0[2]])
	# Dot product
	return np.dot(L, v)

# Let's try with this thing...

def modulated_particle(x0, y0, T, epsilon):
	#print("tracking particle ({},{})".format(x0,y0))
	v = np.array([x0, 0., y0, 0.])
	for i in range(T):
		v = modulated_henon_map(v, epsilon, i)
		if np.absolute(v[0]) > 1000 or np.absolute(v[2]) > 1000:
			# particle lost!
			#print("particle ({},{}) lost at step {}.".format(x0,y0,i))
			return i
	# particle not lost!
	#print("particle ({},{}) survived.".format(x0,y0))
	return -1

def modulated_map_scan(x0, x1, y0, y1, resx, resy, T, epsilon):
	'''
	Tracks all the particles in a rectangular zone
	'''
	region = [[modulated_particle((x/resx) * (x1-x0) + x0, (y/resy) * (y1-y0) + y0, T, epsilon) for y in range(resy)] for x in range(resx)]
	return np.asarray(region)

def modulated_radius_scan(theta, dx, T, epsilon, stop_condition = "first_unstable", boundary = 1000):
	'''
	Given the angle and the radial step:
	> if "first_unstable" will stop at the first (returns last stable step)
	> if "boundary" will stop ad boundary (returns the whole array)
	'''
	if stop_condition == "first_unstable":
		i = -1
		flag = True
		while flag:
			i += 1
			flag = modulated_particle(i * dx * np.cos(theta), i * dx * np.sin(theta), T, epsilon) == -1
		return i * dx

	elif stop_condition == "boundary":
		results = np.empty((boundary))
		for i in boundary:
			results[i] = modulated_particle(i * dx * np.cos(theta), i * dx * np.sin(theta), T, epsilon)
		return np.asarray(results)
	else:
		print("Error: stop condition not contemplated!")
		assert False
