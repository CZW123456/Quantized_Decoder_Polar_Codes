from ConventionalDecoder.CodeConstruction import phi_inverse,  phi
import numpy as np
from QuantizeDecoder.OptUniformQuantizerGaussian import OptUniformQuantizerGaussian

class LLRLSUniformQuantizer():

    def __init__(self, N, v):
        self.N = N
        self.v = v

    def generate_uniform_quantizers(self, sigma):
        n = int(np.log2(self.N))
        mu_llr = np.zeros((n + 1, self.N))
        mu_llr[0, :] = 2 / sigma ** 2
        Q =  OptUniformQuantizerGaussian(self.v)
        decoder_r_f = np.zeros(shape=self.N-1)
        decoder_r_g = np.zeros(shape=self.N-1)
        for level in range(1, n + 1):
            num_bits_parent_node = 2 ** (n - level + 1)
            num_bits_cur_node = num_bits_parent_node // 2
            num_node_parent_level = 2 ** (level - 1)
            for node in range(num_node_parent_level):
                lnode = 2 * node
                rnode = 2 * node + 1
                pnode_posi = 2**(level-1) - 1 + node
                poffset = num_bits_parent_node * node
                mu = mu_llr[level - 1, poffset]
                mu_f = phi_inverse(1 - (1 - phi(mu)) ** 2)
                opt_r_f = Q.find_optimal_interval_bimodal_Gaussian(mu=mu_f, sigma2=2*mu_f, max_iter=30)
                mu_g = 2 * mu
                opt_r_g = Q.find_optimal_interval_bimodal_Gaussian(mu=mu_g, sigma2=2 * mu_g, max_iter=30)
                mu_llr[level, lnode * num_bits_cur_node:(lnode + 1) * num_bits_cur_node] = mu_f
                mu_llr[level, rnode * num_bits_cur_node:(rnode + 1) * num_bits_cur_node] = mu_g
                decoder_r_f[pnode_posi] = opt_r_f
                decoder_r_g[pnode_posi] =  opt_r_g
        return decoder_r_f, decoder_r_g

if __name__ == "__main__":
    N = 128
    v = 32
    sigma2_awgn = 1
    mu_llr = 2/sigma2_awgn
    sigma2_llr = 2 * mu_llr
    quantizer = LLRLSUniformQuantizer(N, v)
    decoder_r_f, decoder_r_g = quantizer.generate_uniform_quantizers(sigma2_llr)
    print()





