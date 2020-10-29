from ConventionalDecoder.CodeConstruction import phi_inverse,  phi
import numpy as np
from QuantizeDecoder.LloydQuantizer import LloydQuantizer
from tqdm import tqdm

class LLRLloydGA():

    def __init__(self, N, v, max_iter=30):
        self.N = N
        self.v = v
        self.max_iter = max_iter

    def generate_Lloyd_quantizers(self, sigma):
        n = int(np.log2(self.N))
        mu_llr = np.zeros((n + 1, self.N))
        mu_llr[0, :] = 2 / sigma ** 2
        Q =  LloydQuantizer(K=self.v, max_iter=self.max_iter)
        decoder_boundary_f = np.zeros(shape=(self.N - 1, self.v + 1))
        decoder_boundary_g = np.zeros(shape=(self.N - 1, self.v + 1))
        decoder_reconstruct_f = np.zeros(shape=(self.N - 1, self.v))
        decoder_reconstruct_g = np.zeros(shape=(self.N - 1, self.v))
        for level in tqdm(range(1, n + 1)):
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
                opt_boundary_f, opt_reconstruct_f = Q.find_quantizer_gaussian(mu_f, 2*mu_f,
                                                                              begin=-mu_f - 3 * np.sqrt(2*mu_f),
                                                                              end=mu_f + 3 * np.sqrt(2*mu_f))
                mu_g = 2 * mu
                opt_boundary_g, opt_reconstruct_g = Q.find_quantizer_gaussian(mu_g, 2*mu_g,
                                                                              begin=-mu_g - 3 * np.sqrt(2*mu_g),
                                                                              end=mu_g + 3 * np.sqrt(2*mu_g))
                mu_llr[level, lnode * num_bits_cur_node:(lnode + 1) * num_bits_cur_node] = mu_f
                mu_llr[level, rnode * num_bits_cur_node:(rnode + 1) * num_bits_cur_node] = mu_g
                decoder_boundary_f[pnode_posi, :] = opt_boundary_f
                decoder_reconstruct_f[pnode_posi, :] = opt_reconstruct_f
                decoder_boundary_g[pnode_posi, :] = opt_boundary_g
                decoder_reconstruct_g[pnode_posi, :] = opt_reconstruct_g

        return decoder_boundary_f, decoder_boundary_g, decoder_reconstruct_f, decoder_reconstruct_g

if __name__ == "__main__":
    N = 16
    v = 16
    sigma2_awgn = 1
    mu_llr = 2/sigma2_awgn
    sigma2_llr = 2 * mu_llr
    quantizer = LLRLloydGA(N, v)
    decoder_boundary_f, decoder_boundary_g, decoder_reconstruct_f, decoder_reconstruct_g = quantizer.generate_Lloyd_quantizers(sigma2_llr)
    print()




