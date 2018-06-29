import numpy as np
import matplotlib.pyplot as plt

iterations = 1000
side = 100
lenght = 0.6
mu = 1
ni_x = 0.28
ni_y = 0.31
boundary = 10.0

'''
def henon_map(v, n):
	# v is the quadrimentional vector (x, px, y, py)
	# n is the iteration number
	# Omegas computation
	omega_x = ni_x * 2 * np.pi
	omega_y = ni_y * 2 * np.pi
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
	v = np.array([v[0],
				  v[1] + v[0]*v[0] - v[2]*v[2] + mu*(-3*v[2]*v[2]*v[0] + v[0]*v[0]*v[0]),
				  v[2],
				  v[3] - 2*v[0]*v[2] + mu*(+3*v[0]*v[0]*v[2] - v[2]*v[2]*v[2])])
	# Dot product
	return np.dot(L, v)
'''
def henon_map(v, n):
	omega_x = ni_x * 2 * np.pi
	omega_y = ni_y * 2 * np.pi
	cosx = np.cos(omega_x)
	sinx = np.sin(omega_x)
	cosy = np.cos(omega_y)
	siny = np.sin(omega_y)
	return np.array([cosx * v[0] + sinx * (v[1] + (v[0] * v[0] - v[2] * v[2]) + mu * (v[0] * v[0] * v[0] - 3 * v[0] * v[2] * v[2])),
		- sinx * v[0] + cosx * (v[1] + (v[0] * v[0] - v[2] * v[2]) + mu * (v[0] * v[0] * v[0] - 3 * v[0] * v[2] * v[2])),
		cosy * v[2] + siny * (v[3] - 2 * v[0] * v[2] + mu * (3 * v[0] * v[0] * v[2] - v[2] * v[2] * v[2])),
		- siny * v[2] + cosy * (v[3] - 2 * v[0] * v[2] + mu * (3 * v[0] * v[0] * v[2] - v[2] * v[2] * v[2]))])

def particle(x0, y0):
	print("tracking particle ({},{})".format(x0, y0))
	v = np.array([x0, 0., y0, 0.])
	for i in range(iterations):
		temp = henon_map(v, i)
		v = temp
		if v[0]*v[0] +v[1]*v[1] +v[2]*v[2] +v[3]*v[3] > 1e6:
			# particle lost!
			#print("particle ({},{}) lost at step {}.".format(x0,y0,i))
			return i
	# particle not lost!
	#print("particle ({},{}) survived.".format(x0,y0))
	return -1

graph = [[0 if particle((x/side) * lenght, (y/side) * lenght) != -1 else 1 for y in range(side)] for x in range(side)]

np.save("graph", graph)

plt.imshow(graph, origin='lower', extent = (0, 0.6, 0, 0.6))
plt.savefig("map_l{}_mu{}_vx{}_vy{}_t{}_bound{}.png".format(lenght,mu,ni_x,ni_y,iterations,boundary))