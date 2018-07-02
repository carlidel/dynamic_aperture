from tracker import *

# "The step in r must be equal to the step
# in theta times <|dr/dtheta|> to optimize
# the integration steps."

# TODO :: elaborate new method of stability map display

'''
look = modulated_map_scan(-1,1,0,1,200,100,1000,1)
look[look != -1] = 0
look[look == -1] = 1
print(look)
plt.imshow(look, origin='lower', extent = (0, 1, -1, 1))
plt.show()
'''
dx = 0.01
nangles = 30
angles = np.linspace(0, np.pi/4, nangles)
epsilons = [0, 1]
n_turns = [2000, 7000, 50000, 200000, 700000]

for t in n_turns:
	for e in epsilons:
		print("Working now on t={}, epsilons={}...".format(t,e))
		data = np.empty((len(angles)))
		for i in range(len(angles)):
			data[i] = modulated_radius_scan(angles[i], dx, t, e)
		np.save("radscan_T{}_dx{}_nthet{}_epsilon{}".format(t,dx,nangles,e), data)

