from tracker import *
import matplotlib.pyplot as plt

look = modulated_map_scan(0,1,0,1,200,10,1000,1)
look[look != -1] = 0
look[look == -1] = 1
plt.imshow(look, origin='lower', extent = (-1, 1, 0, 1))
plt.show()
