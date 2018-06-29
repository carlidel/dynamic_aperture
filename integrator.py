import numpy as np

survival1000 = np.load("survival_limits_time1000.npy")
n_theta = 30 # Number of scanned angles
values = [0, 1, 4, 16, 64]
times = [1000, 100000, 10000000]
angles = np.linspace(0., np.pi/2, num = n_theta)

j=0
for line in survival1000:
	D = 0
	#print(line)
	for i in range(0,n_theta-2,2):
					f1 = (line[ i ]**4)*np.sin(2*angles[i])
					f2 = (line[i+1]**4)*np.sin(2*angles[i+1])
					f3 = (line[i+2]**4)*np.sin(2*angles[i+2])
					D += ((angles[i+2]-angles[i])/6)*(f1 + 4 * f2 + f3)
			
	D = D**(1/4)
	print(values[j], D)
	#print(np.average(line))
	j+=1