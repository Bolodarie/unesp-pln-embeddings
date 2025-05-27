import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("vecs.dat")

plt.plot(data,lw=0,marker='o')
plt.show()