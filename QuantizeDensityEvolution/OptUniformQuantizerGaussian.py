import numpy as np
from scipy.integrate import quad
from scipy.special import erf
import matplotlib.pyplot as plt

class OptUniformQuantizerGaussian():

    def __init__(self, K):
        self.K = K

    def gaussian(self, mu, sigma2, x):
        return 1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-(x - mu) ** 2 / (2 * sigma2))

    def intergral_f_px(self, a, b, reconstruction_value):
        f = lambda x: (x - reconstruction_value)**2 / (np.sqrt(2 * np.pi * self.sigma2)) * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma2))
        return quad(f, a, b)[0]

    def compute_distortion(self, r):
        distortion = 0
        N = self.K//2
        for i in range(1, N):
            reconstruction_value = (2 * i - 1) * r / 2
            begin = (i - 1) * r
            end = i * r
            distortion += self.intergral_f_px(begin, end, reconstruction_value)
        distortion += self.intergral_f_px((N - 1)*r, np.inf, (N - 1/2) * r)
        return distortion


    def d2r_middle(self, i, a, b, r):
        c1 = 1/np.sqrt(2*np.pi*self.sigma2)
        c2 = self.mu - r
        mu = self.mu
        sigma2 = self.sigma2
        part1 = c1 * (i*(b-mu)*np.exp(-(b-mu)**2/(2*np.pi*sigma2)) - (i-1)*(a-mu)*np.exp(-(a-mu)**2/(2*sigma2)))
        part2 = c2*c1*(i*np.exp(-(b-mu)**2/(2*np.pi*sigma2)) - (i-1)*np.exp(-(a-mu)**2/(2*sigma2)))
        return 2 * (part1 + part2)

    def d2r_last(self, i, a, r):
        c1 = 1 / np.sqrt(2 * np.pi * self.sigma2)
        c2 = self.mu - r
        mu = self.mu
        sigma2 = self.sigma2
        part1 = -c1 * ((i - 1) * (a - mu) * np.exp(-(a - mu) ** 2 / (2 * sigma2)))
        part2 = -c2 * c1 * ((i - 1) * np.exp(-(a - mu) ** 2 / (2 * sigma2)))
        return 2 * (part1 + part2)

    def d2r(self, r):
        result = 0
        N = self.K // 2
        for k in range(1, N):
            result -= (2 * k - 1) * self.d2r_middle(k, (k - 1) * r, k * r, (k - 1 / 2) * r)
        result -= (2 * N - 1) * self.d2r_last(N, (N - 1) * r, (N - 1 / 2) * r)
        return result

    def dr_middle_unimodal_Gaussian(self, a, b, r):
        return 2 * (-sigma2 * (self.gaussian(self.mu, self.sigma2, b) - self.gaussian(self.mu, self.sigma2, a)) + 0.5 * (self.mu - r) * (erf((b-self.mu)/(np.sqrt(2*sigma2))) - erf((a-self.mu)/(np.sqrt(2*sigma2)))))

    def dr_last_unimodal_Gaussian(self, a, r):
        return 2 * (self.sigma2 * self.gaussian(self.mu, self.sigma2, a) + 0.5 * (self.mu - r) * (1 - erf((a - self.mu) / (np.sqrt(2 * self.sigma2)))))

    def dr_unimodal_Gaussian(self, r):
        result = 0
        N = self.K//2
        for k in range(1, N):
            result -= (2*k-1) * self.dr_middle_unimodal_Gaussian((k-1)*r, k*r, (k-1/2)*r)
        result -= (2*N - 1) * self.dr_last_unimodal_Gaussian((N - 1)*r, (N - 1/2)*r)
        return result

    def dr_middle_bimodal_Gaussian(self, a, b, r):
        mu = self.mu
        inv_mu = -self.mu
        sigma2 = self.sigma2
        part1 = -sigma2/2 * (self.gaussian(mu, sigma2, b) - self.gaussian(mu, sigma2, a))
        part2 = 0.25 * (mu - r) * (erf((b-mu)/(np.sqrt(2*sigma2))) - erf((a-mu)/(np.sqrt(2*sigma2))))
        part3 = -sigma2/2 * (self.gaussian(inv_mu, sigma2, b) - self.gaussian(inv_mu, sigma2, a))
        part4 = 0.25 * (-mu - r) * (erf((b-inv_mu)/(np.sqrt(2*sigma2))) - erf((a-inv_mu)/(np.sqrt(2*sigma2))))
        return 2 * (part1 + part2 + part3 + part4)

    def dr_last_bimodal_Gaussian(self, a, r):
        mu = self.mu
        inv_mu = -self.mu
        sigma2 = self.sigma2
        part1 = sigma2/2 * self.gaussian(mu, sigma2, a) + 0.25 * (mu - r) * (1 - erf((a - mu) / (np.sqrt(2 * sigma2))))
        part2 = sigma2/2 * self.gaussian(inv_mu, sigma2, a) + 0.25 * (inv_mu - r) * (1 - erf((a - inv_mu) / (np.sqrt(2 * sigma2))))
        return 2 * (part1 + part2)

    def dr_bimodal_Gaussian(self, r):
        result = 0
        N = self.K//2
        for k in range(1, N):
            result -= (2*k-1) * self.dr_middle_bimodal_Gaussian((k-1)*r, k*r, (k-1/2)*r)
        result -= (2 * N - 1) * self.dr_last_bimodal_Gaussian((N - 1) * r, (N - 1 / 2) * r)
        return result


    def numerical_d2r_unimodal_Gaussian(self, r, delta):
        return (self.dr_unimodal_Gaussian(r + delta) - self.dr_unimodal_Gaussian(r - delta)) / (2 * delta)


    def numerical_d2r_bimodal_Gaussian(self, r, delta):
        return (self.dr_bimodal_Gaussian(r + delta) - self.dr_bimodal_Gaussian(r - delta)) / (2 * delta)

    def find_optimal_interval_unimodal_Gaussian(self, mu, sigma2, max_iter=30):
        self.mu = mu
        self.sigma2 = sigma2
        # initial r
        interval_length = self.mu + 3 * np.sqrt(self.sigma2)
        r = 2 * interval_length / (self.K - 2)
        opt_r = r
        for i in range(max_iter):
            d2r = self.numerical_d2r_unimodal_Gaussian(opt_r, 1e-6)
            dr = self.dr_unimodal_Gaussian(opt_r)
            opt_r -= dr/d2r
            dr_new = self.dr_unimodal_Gaussian(opt_r)
            if np.abs(dr_new - dr) < 1e-6:
                break
            print("iter = {:d}, dr = {:f}".format(i, dr_new))
        return opt_r

    def find_optimal_interval_bimodal_Gaussian(self, mu, sigma2, max_iter=30):
        self.mu = mu
        self.sigma2 = sigma2
        # initial r
        interval_length = mu+3*np.sqrt(sigma2) - (-mu-3*np.sqrt(sigma2))
        r = 2 * interval_length / (self.K - 2)
        opt_r = r
        for i in range(max_iter):
            d2r = self.numerical_d2r_bimodal_Gaussian(opt_r, 1e-6)
            dr = self.dr_bimodal_Gaussian(opt_r)
            opt_r -= dr/d2r
            dr_new = self.dr_bimodal_Gaussian(opt_r)
            if np.abs(dr_new - dr) < 1e-6:
                break
            # print("iter = {:d}, dr = {:f}".format(i, dr_new))
        return opt_r

if __name__ == "__main__":
    sigma2 = 0.79
    mu = 2/sigma2
    K = 32
    max_iter = 200
    Q = OptUniformQuantizerGaussian(K)
    opt_r_bimodal = Q.find_optimal_interval_bimodal_Gaussian(mu, sigma2, max_iter)
    # print("Optimal interval for unimodal Gaussian distribution = {:f}".format(opt_r_unimodal))
    print("Optimal interval for bimodal Gaussian distribution = {:f}".format(opt_r_bimodal))

    # plot optimal quantization for bimodal Gaussian distribution
    x_bimodal = np.linspace(mu+3*np.sqrt(sigma2), -mu-3*np.sqrt(sigma2), num=1000)
    px_bimodal = 0.5 * (1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-(x_bimodal - mu) ** 2 / (2 * sigma2)) +
                 1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-(x_bimodal + mu) ** 2 / (2 * sigma2)))
    plt.figure(1)
    plt.plot(x_bimodal, px_bimodal)
    for i in range(K // 2):
        plt.vlines(x=i * opt_r_bimodal, ymin=0, ymax=np.max(px_bimodal), linestyles="dashed")
        plt.vlines(x=-i * opt_r_bimodal, ymin=0, ymax=np.max(px_bimodal), linestyles="dashed")
    plt.show()




