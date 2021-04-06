from scipy.special import erf
import numpy as np
from PyIBQuantizer.inf_theory_tools import log_stable


class LLRMIQuantizer():

    def __init__(self, K):
        self.K = K


    def compute_mutual_information(self, d, r, mu, sigma2):
        I = 0
        for i in range(self.K):
            p1 = erf((d[i+1]-mu)/np.sqrt(2*sigma2)) - erf((d[i]-mu)/np.sqrt(2*sigma2))
            p2 = erf((d[i+1]+mu)/np.sqrt(2*sigma2)) - erf((d[i]+mu)/np.sqrt(2*sigma2))
            I += (1 - np.log2(1+np.exp(-r[i]))) * p1 + (1 - np.log2(1+np.exp(r[i]))) * p2
        I /= 4
        return I


    def compute_mutual_information_discrete(self, plx, r):
        I = np.sum((1 - np.log2(1+np.exp(-r)))*plx[0] + (1 - np.log2(1+np.exp(r)))*plx[1])
        return I / 2


    def compute_partial_mutual_information(self, M, K, plx):
        table = np.zeros((M, M + 1))
        for a_prime in range(M):
            max_a = np.min([a_prime + M - K + 1, M])
            for a in range(a_prime + 1, max_a + 1):
                r_i = log_stable(np.sum(plx[0, a_prime:a])/(np.sum(plx[1, a_prime:a])))
                table[a_prime, a] = 0.5 * np.sum((1 - np.log2(1 + np.exp(-r_i))) * plx[0, a_prime:a] + (1 - np.log2(1 + np.exp(r_i))) * plx[1, a_prime:a])
        return table


    def find_quantizer_awgn(self, mu, sigma2, begin, end):
        d = np.zeros(self.K + 1)
        d[0] = -np.inf
        d[-1] = np.inf
        d[1:-1] = np.linspace(start=begin, stop=end, num=self.K-1)
        r = np.zeros(self.K)
        r[0] = d[1] - 2
        for i in range(1, self.K - 1):
            r[i] = d[i] + (d[i + 1] - d[i]) / 2
        r[-1] = d[-2] + 2
        for iter in range(200):
            I_before = self.compute_mutual_information(d, r, mu, sigma2)
            # update r when d is fixed
            for i in range(self.K):
                p1 = erf((d[i+1] - mu)/np.sqrt(2*sigma2)) - erf((d[i] - mu)/np.sqrt(2*sigma2))
                p2 = erf((d[i+1] + mu)/np.sqrt(2*sigma2)) - erf((d[i] + mu)/np.sqrt(2*sigma2))
                r[i] = np.log(p1/p2)
            # update d when r is fixed
            for i in range(1, self.K):
                c1 = np.log2((1+np.exp(-r[i]))/(1+np.exp(-r[i-1])))
                c2 = np.log2((1+np.exp(r[i]))/(1+np.exp(r[i-1])))
                d[i] = sigma2 * (np.log(c2) - np.log(-c1)) / (2 * mu)
            I_after = self.compute_mutual_information(d, r, mu, sigma2)
            if I_after - I_before < 1e-6:
                break
        p = np.zeros((2, self.K))
        for i in range(self.K):
            p[0, i] = 0.5 * (erf((d[i+1] - mu)/np.sqrt(2*sigma2)) - erf((d[i] - mu)/np.sqrt(2*sigma2)))
            p[1, i] = 0.5 * (erf((d[i+1] + mu)/np.sqrt(2*sigma2)) - erf((d[i] + mu)/np.sqrt(2*sigma2)))
        return d, r, p


    def find_quantizer_decoder(self, plx, llr, M, K):
        assert M > K
        # permutation = np.argsort(llr)
        llr = np.log(plx[0]/plx[1])
        permutation = np.argsort(llr)
        tmp_plx = np.zeros_like(plx)
        tmp_plx[0, :] = plx[0, permutation]
        tmp_plx[1, :] = plx[1, permutation]
        self.partial_entropy_table = self.compute_partial_mutual_information(M, K, tmp_plx)
        # print(self.partial_entropy_table)
        Az = np.zeros(K + 1, dtype=int)
        Az[-1] = M
        state_table = np.zeros((M - K + 1, K + 1))
        local_max = -np.ones_like(state_table) * np.inf
        state_table[:, 1] = self.partial_entropy_table[0, 1:M - K + 2]
        local_max[:, 1] = 0

        # forward computing
        for z in range(2, K + 1):
            if z < K:
                for a in range(z, z + M - K + 1):
                    a_idx = a - z
                    a_prime_begin = z - 1
                    a_prime_end = a - 1
                    tmp = np.zeros(a_prime_end - a_prime_begin + 1)
                    tmp_idx = []
                    cnt = 0
                    for a_prime in range(a_prime_begin, a_prime_end + 1):
                        tmp[cnt] = state_table[a_prime - a_prime_begin, z - 1] + self.partial_entropy_table[a_prime, a]
                        tmp_idx.append(a_prime)
                        cnt += 1
                    local_max[a_idx, z] = tmp_idx[int(np.argmax(tmp))]
                    state_table[a_idx, z] = tmp[int(np.argmax(tmp))]
            else:
                a = M
                a_prime_begin = z - 1
                a_prime_end = a - 1
                tmp = np.zeros(a_prime_end - a_prime_begin + 1)
                tmp_idx = []
                cnt = 0
                for a_prime in range(a_prime_begin, a_prime_end + 1):
                    tmp[cnt] = state_table[a_prime - a_prime_begin, z - 1] + self.partial_entropy_table[a_prime, a]
                    tmp_idx.append(a_prime)
                    cnt += 1
                state_table[-1, z] = np.max(tmp)
                local_max[-1, z] = tmp_idx[np.argmax(tmp)]

        # backward tracing
        Az[-2] = int(local_max[-1, -1])
        opt_idx = int(local_max[-1, -1])
        for z in range(K - 1, 1, -1):
            opt_idx = int(local_max[opt_idx - z, z])  #
            Az[z - 1] = int(opt_idx)

        # LUT generation
        lut = np.zeros(M).astype(np.int32)
        llr_quanta = np.zeros(K).astype(np.float64)
        plx_compressed = np.zeros((2, K)).astype(np.float64)
        for i in range(K):
            begin = Az[i]
            end = Az[i + 1]
            lut[permutation[begin:end]] = i
            llr_quanta[i] = log_stable(np.sum(plx[0, permutation[begin:end]])/np.sum(plx[1, permutation[begin:end]]))
            plx_compressed[0, i] = np.sum(plx[0, permutation[begin:end]])
            plx_compressed[1, i] = np.sum(plx[1, permutation[begin:end]])
        return plx_compressed, llr_quanta, lut, state_table[-1 ,-1], tmp_plx, self.partial_entropy_table


if __name__ == "__main__":
    sigma2_awgn = 1
    mu = 2/sigma2_awgn
    sigma2_llr = 2*mu
    begin = -mu - 3*np.sqrt(sigma2_llr)
    end = mu + 3*np.sqrt(sigma2_llr)
    quantizer = LLRMIQuantizer(16)
    quantizer.find_quantizer_awgn(mu, sigma2_llr, begin, end)

