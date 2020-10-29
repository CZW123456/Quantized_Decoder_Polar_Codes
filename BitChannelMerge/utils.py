import numpy as np
from scipy.stats import norm

def Clambda(x):
    z = 1 - x/(1+x)*np.log2(1+1/x) - 1/(1+x)*np.log2(1+x)
    return z

def dClambdadx(x):
    z = np.log2(x)/(1+x)**2
    return z

def get_Clambda_zero_points(v):
    alpha = np.zeros(v + 1)
    alpha[0] = 1
    alpha[-1] = 1e308
    eps = 1e-6
    for i in range(1, v):
        beta = i / v
        x0 = 0
        x1 = 1.5
        while(np.abs(x0 - x1) > eps):
            x0 = x1
            x1 = x0 - (Clambda(x0) - beta) / dClambdadx(x0)
        alpha[i] = x1
    return alpha

def get_y_interval(sigma, alpha):
    y = np.zeros((alpha.shape[0]-1, 2))
    for i in range(alpha.shape[0] - 1):
        y[i, 0] = sigma**2/2*np.log(alpha[i]) # loge!
        y[i, 1] = sigma**2/2*np.log(alpha[i+1])
    return y

def degrading_transform_AWGN_to_DMC(y, sigma, v):
    W = np.zeros((2, 2 * v))
    for i in range(v):
        y_min = y[i, 0]
        y_max = y[i, 1]
        p0 = norm.cdf(y_max, 1, sigma) - norm.cdf(y_min, 1, sigma)
        p1 = norm.cdf(y_max, -1, sigma) - norm.cdf(y_min, -1, sigma)
        W[0, 2*i] = p0
        W[1, 2*i] = p1
        W[0, 2*i+1] = p1
        W[1, 2*i+1] = p0
    return W

def upgrading_transform_AWGN_to_DMC(y, theta, sigma, v):
    W = np.zeros((2, 2 * v))
    for i in range(v):
        if i < v - 1:
            y_min = y[i, 0]
            y_max = y[i, 1]
            p0 = norm.cdf(y_max, 1, sigma) - norm.cdf(y_min, 1, sigma)
            p1 = norm.cdf(y_max, -1, sigma) - norm.cdf(y_min, -1, sigma)
            pi_i = p0 + p1
            z0 = (theta[i+1]*pi_i)/(1+theta[i+1])
            z1 = pi_i/(1+theta[i+1])
            W[0, 2*i] = z0
            W[1, 2*i] = z1
            W[0, 2*i+1] = z1
            W[1, 2*i+1] = z0
        else:
            y_min = y[i, 0]
            y_max = y[i, 1]
            p0 = norm.cdf(y_max, 1, sigma) - norm.cdf(y_min, 1, sigma)
            p1 = norm.cdf(y_max, -1, sigma) - norm.cdf(y_min, -1, sigma)
            pi_i = p0 + p1
            W[0, 2 * i] = pi_i
            W[1, 2 * i] = 0
            W[0, 2 * i + 1] = 0
            W[1, 2 * i + 1] = pi_i
    return W

def get_AWGN_transition_probability_degrading(sigma, v):
    alpha = get_Clambda_zero_points(v)
    y = get_y_interval(sigma, alpha)
    W = degrading_transform_AWGN_to_DMC(y, sigma, v)
    y_interval_posi = np.zeros((v + 1))
    y_interval = np.zeros(2 * v + 1)
    for i in range(v):
        y_interval_posi[i] = y[i, 0]
        y_interval_posi[i + 1] = y[i, 1]
    y_interval[0:v + 1] = -y_interval_posi[::-1]
    y_interval[v + 1:] = y_interval_posi[1:]
    lr = W[0] / W[1]
    indices = np.argsort(lr)
    W = W[:, indices]
    return W, y_interval

def get_AWGN_transition_probability_upgrading(sigma, v):
    alpha = get_Clambda_zero_points(v)
    y = get_y_interval(sigma, alpha)
    W = upgrading_transform_AWGN_to_DMC(y, alpha, sigma, v)
    y_interval_posi = np.zeros((v+1))
    y_interval = np.zeros(2*v+1)
    for i in range(v):
        y_interval_posi[i] = y[i, 0]
        y_interval_posi[i+1] = y[i, 1]
    y_interval[0:v+1] = -y_interval_posi[::-1]
    y_interval[v+1:] = y_interval_posi[1:]
    lr = W[0] / W[1]
    indices = np.argsort(lr)
    W = W[:, indices]
    return W, y_interval

def check_symmetric(W):
    N = W.shape[1]
    for i in range(N//2):
        if W[0, i] != W[1, N-1-i] or W[1, i] != W[0, N-1-i]:
            return False
    return True