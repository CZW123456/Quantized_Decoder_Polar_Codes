from PyIBQuantizer.modified_sIB import modified_sIB
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

a = np.array([0.56394366, 0.24777228, 0.05468451, 0.00907574])
b = np.array([0.590909, 0.240509, 0.056287, 0.009407])

plt.figure()
plt.semilogy(a)
plt.semilogy(b)
plt.show()

