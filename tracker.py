import numpy as np

file = open("Data.txt", "w") 
file.write("Begin!")

# Parameters
dx = 1.e-2 # Delta space
n_theta = 30 # Number of scanned angles

T = 100000 # Number of turns

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

print(epsilon_k)
print(Omega_k)

omega_x0 = 0.168 * 2 * np.pi
omega_y0 = 0.201 * 2 * np.pi

epsilon = 1 

boundary = 1.0

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

'''
def binary_search(low, high, theta):
	middle = (low + high) // 2
	#print(low, middle, high)
	if middle == low:
		return (False, middle, middle)
	if particle(middle * dx * np.cos(theta), middle * dx * np.sin(theta)) != -1:
		return (True, low, middle)
	else:
		return (True, middle, high)
'''
values = [0, 1, 4, 16, 64]
times = [1000]#, 10000, 100000]
angles = np.linspace(0., np.pi / 2, num=n_theta)
survival_limits = []

for time in times:
	print(time)
	T = time
	temp = []
	for value in values:
		print(value)
		epsilon = value
		temp2 = []
		for angle in angles:
			i = 0
			flag = True
			while flag:
				flag = particle(i * dx * np.cos(angle), i * dx * np.sin(angle)) == -1
				i += 1
			temp2.append(i * dx)
		temp.append(temp2)
	survival_limits.append(temp)
	print(temp)
	np.save("survival_limits_time{}".format(time), temp)

survival_limits = np.asarray(survival_limits)
print(survival_limits)
np.save("survival_limits", survival_limits)