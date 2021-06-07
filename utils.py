from bisect import bisect_left
import numpy as np

def continous2discret(x, xs, max_level):
    if x <= xs[0]:
        return 0
    if x >= xs[-1]:
        return max_level

    i = bisect_left(xs, x)   # binary search for determining the smallest number that is larger than x

    return int(i - 1)


def channel_transition_probability_table(M, low, high, mu, sigma):
    delta = 0.0001
    x_continuous = np.arange(low, high + delta, delta)
    pyx_continuous = 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x_continuous - mu)**2 / (2*sigma**2))
    x_discrete = np.linspace(low, high, M + 1)
    pyx = np.zeros(M)
    for i in range(M):
        index1 = x_continuous >= x_discrete[i]
        index2 = x_continuous <= x_discrete[i + 1]
        index = np.bitwise_and(index1, index2)
        density = pyx_continuous[index]
        pyx[i] = np.sum(density) * delta
    return pyx, x_discrete


def channel_llr_density_table(M, low, high, mu1, mu2, sigma):
    delta = 0.0001
    x_continuous = np.arange(low, high + delta, delta)
    pyx_continuous = 0.5*(1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x_continuous-mu1)**2/(2*sigma**2)) +
                          1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x_continuous-mu2)**2/(2*sigma**2)))
    x_discrete = np.linspace(low, high, M + 1)
    quanta = np.zeros(M)
    pyx = np.zeros(M)
    for i in range(M):
        index1 = x_continuous >= x_discrete[i]
        index2 = x_continuous <= x_discrete[i + 1]
        index = np.bitwise_and(index1, index2)
        density = pyx_continuous[index]
        pyx[i] = np.sum(density) * delta
        quanta[i] = np.sum(x_continuous[index] * density) / np.sum(density)
    return pyx, x_discrete, quanta