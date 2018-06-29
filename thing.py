import numpy as np
import matplotlib.pyplot as plt

iterations = 1000
side = 100
lenght = 0.6
mu = 1
ni_x = 0.28
ni_y = 0.31
boundary = 10.0

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

def particle(x0, y0):
	print("tracking particle ({},{})".format(x0, y0))
	v = np.array([x0, 0., y0, 0.])
	
	for i in range(4):
		temp = henon_map(v, i)
		v = temp
		print(i,v)

	return -1

particle(0.3, 0.0)