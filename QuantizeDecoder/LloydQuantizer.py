import numpy as np
from scipy.integrate import quad
from scipy.special import erf

class LloydQuantizer():

    def __init__(self, K, max_iter):
        self.K = K
        self.max_iter=  max_iter

    def compute_mutual_information(self, d, r, mu, sigma2):
        I = 0
        for i in range(self.K):
            p1 = erf((d[i+1]-mu)/np.sqrt(2*sigma2)) - erf((d[i]-mu)/np.sqrt(2*sigma2))
            p2 = erf((d[i+1]+mu)/np.sqrt(2*sigma2)) - erf((d[i]+mu)/np.sqrt(2*sigma2))
            I += (1 - np.log2(1+np.exp(-r[i]))) * p1 + (1 - np.log2(1+np.exp(r[i]))) * p2
        I /= 4
        return I

    def compute_total_distortion(self, d, r, mu, sigma2):
        distortion = 0
        for i in range(self.K):
            f = lambda x : (x - r[i])**2 * 0.5 * (1/np.sqrt(2*np.pi*sigma2)*np.exp(-(x-mu)**2/(2*sigma2)) +
                                                  1/np.sqrt(2*np.pi*sigma2)*np.exp(-(x+mu)**2/(2*sigma2)))
            distortion += quad(f, d[i], d[i+1])[0]
        return distortion

    def mass_center_nominator(self, a, b, mu, sigma2):
        f = lambda x : 0.5 * x * (1/np.sqrt(2*np.pi*sigma2)*np.exp(-(x-mu)**2/(2*sigma2)) +
                                  1/np.sqrt(2*np.pi*sigma2)*np.exp(-(x+mu)**2/(2*sigma2)))
        return quad(f, a, b)[0]

    def mass_center_denominator(self, a, b, mu, sigma2):
        f = lambda x : 0.5 * (1/np.sqrt(2*np.pi*sigma2)*np.exp(-(x-mu)**2/(2*sigma2)) +
                              1/np.sqrt(2*np.pi*sigma2)*np.exp(-(x+mu)**2/(2*sigma2)))
        return quad(f, a, b)[0]


    def find_quantizer_gaussian(self, mu_llr, sigma2_llr, begin, end):
        # using uniform quantization of the 3\sigma interval as initialization
        d = np.zeros(self.K + 1)
        d[0] = -np.inf
        d[-1] = np.inf
        d[1:-1] = np.linspace(start=begin, stop=end, num=self.K-1)
        r = np.zeros(self.K)
        r[0] = d[1] - 2
        for i in range(1, self.K - 1):
            r[i] = d[i] + (d[i + 1] - d[i]) / 2
        r[-1] = d[-2] + 2
        for iter in range(self.max_iter):
            distortion_before = self.compute_total_distortion(d, r, mu_llr, sigma2_llr)
            # fix d then optimizing r
            for i in range(self.K):
                nominator = self.mass_center_nominator(d[i], d[i+1], mu_llr, sigma2_llr)
                denominator = self.mass_center_denominator(d[i], d[i+1], mu_llr, sigma2_llr)
                r[i] = nominator / denominator
            # fix r then optimizing d
            for i in range(1, self.K):
                d[i] = 0.5 * (r[i-1] + r[i])
            # compute total distortion after the this run of optimization
            distortion_after = self.compute_total_distortion(d, r, mu_llr, sigma2_llr)
            # print("Iter = {:d}, Total Distortion = {:f}".format(iter, distortion_after))
            if distortion_before - distortion_after < 1e-6:
                break
        d[0] = -1e300
        d[-1] = 1e300
        return d, r


if __name__ == "__main__":
    K = 32
    sigma2_awgn = 1
    mu_llr = 2 / sigma2_awgn
    sigma2_llr = 2 * mu_llr
    begin = -mu_llr - 3*np.sqrt(sigma2_llr)
    end = mu_llr + 3*np.sqrt(sigma2_llr)
    Quantizer = LloydQuantizer(K=K, max_iter=1000)
    d, r = Quantizer.find_quantizer_gaussian(mu_llr, sigma2_llr, begin, end)
    I = Quantizer.compute_mutual_information(d, r, mu_llr, sigma2_llr)
    print("Mutual information preserved by the Least Square Quantizer : {:f}".format(I))

