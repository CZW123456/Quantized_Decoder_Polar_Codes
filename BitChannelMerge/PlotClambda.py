import matplotlib.pyplot as plt
import numpy as np

l = np.linspace(1, 1000, 1000)

Cl = 1 - l/(l+1)*np.log2(1+1/l) - 1/(l+1)*np.log2(1+l)

tmp1 = np.log2(1 + l)
tmp2 = np.log2(1+1/l)

plt.figure()
plt.subplot(311)
plt.plot(l, Cl)
plt.subplot(312)
plt.plot(l, tmp1)
plt.subplot(313)
plt.plot(l, tmp2)
plt.show()