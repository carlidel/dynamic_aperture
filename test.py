import numpy as np

iterations = 1000
side = 100
lenght = 0.6
mu = 1
ni_x = 0.28
ni_y = 0.31
boundary = 10.0

def henon_map(v, n):
	'''
	v is the quadrimentional vector (x, px, y, py)
	n is the iteration number
	'''
	# Omegas computation
	omega_x = ni_x * 2 * np.pi
	omega_y = ni_y * 2 * np.pi
	# Linear matrix computation
	cosx = np.cos(omega_x)
	sinx = np.sin(omega_x)
	cosy = np.cos(omega_y)
	siny = np.sin(omega_y)
	print("cosx",cosx)
	print("sinx",sinx)
	print("cosy",cosy)
	print("siny",siny)
	print("solution x", cosx - sinx)
	print("solution y", cosy + siny)
	print("solution px", -sinx -cosx)
	print("solution py", -siny +cosy)
	
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

print(henon_map(np.array([1,1,1,1]),1))
