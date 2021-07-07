import numpy as np
from bisect import bisect_left
from tqdm import tqdm
from PyIBQuantizer.inf_theory_tools import log2_stable
from PyIBQuantizer.modified_sIB import modified_sIB
from PyIBQuantizer.inf_theory_tools import mutual_information

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


class QDensityEvolutionMsIB():

    def __init__(self, N, quantization_level_decoder):
        self.N = N
        self.quantization_level_decoder = quantization_level_decoder

    def get_LUT(self, lut_merge, permutation, mode="f"):
        lut_merge = lut_merge.squeeze()
        if mode == "f":
            lut = np.zeros((self.quantization_level_decoder, self.quantization_level_decoder)).astype(np.int32)
            for i in range(self.quantization_level_decoder):
                indices = np.where(lut_merge == i)[0]
                symbols = permutation[indices]
                y = symbols // self.quantization_level_decoder
                x = symbols % self.quantization_level_decoder
                lut[y, x] = i
            return lut
        else:
            lut = np.zeros((2, self.quantization_level_decoder, self.quantization_level_decoder)).astype(np.int32)
            for i in range(self.quantization_level_decoder):
                indices = np.where(lut_merge == i)[0]
                symbols = permutation[indices]
                u0 = symbols // (self.quantization_level_decoder * self.quantization_level_decoder)
                tmp = symbols - u0 * self.quantization_level_decoder * self.quantization_level_decoder
                y = tmp // self.quantization_level_decoder
                x = tmp % self.quantization_level_decoder
                lut[u0, y, x] = i
            return lut

    def get_border_vector(self, cardY, cardT, num_run):
        alpha = np.ones(int(cardT)) * 1
        border_vectors = np.ones((num_run, cardT)) * cardY
        for run in range(num_run):
            while border_vectors[run, :-1].cumsum().max() >= cardY:
                border_vectors[run] = np.floor(np.random.dirichlet(alpha, 1) * (cardY))
                border_vectors[run, border_vectors[run] == 0] = 1
            border_vectors[run] = np.hstack([border_vectors[run, :-1].cumsum(), cardY]).astype(np.int)
        border_vectors = border_vectors.astype(np.int32)
        return border_vectors


    def run(self, channel_symbol_probs):
        n = int(np.log2(self.N))

        virtual_channel_transition_probs = np.zeros((n, self.N, 2, self.quantization_level_decoder))
        virtual_channel_llrs = np.zeros((n, self.N, self.quantization_level_decoder))

        channel_symbol_probs = np.expand_dims(channel_symbol_probs, axis=0).repeat(self.N, axis=0)

        quantizer = modified_sIB(card_T_=self.quantization_level_decoder, beta=1e30, nror_=1)
        lut_fs = {}
        lut_gs = {}

        for level in tqdm(range(n)):

            num_bits_per_node = 2**(n-level)
            num_lut_per_node = num_bits_per_node // 2
            num_node_cur_level = 2**level

            if level == 0:
                probs = channel_symbol_probs
            else:
                probs = virtual_channel_transition_probs[level - 1]

            for node in range(num_node_cur_level):
                offset = num_bits_per_node * node
                stride = num_bits_per_node // 2
                node_posi = 2**level + node - 1

                P_y0_x0 = probs[offset]
                P_y1_x1 = probs[offset + stride]

                P_y0_x0[0] /= np.sum(P_y0_x0[0])
                P_y0_x0[1] /= np.sum(P_y0_x0[1])
                P_y1_x1[0] /= np.sum(P_y1_x1[0])
                P_y1_x1[1] /= np.sum(P_y1_x1[1])

                # build lut for virtual channel u0 -> y0, y1 -> z0
                P_y0y1_u0_0 = 0.5 * (np.kron(P_y0_x0[0], P_y1_x1[0]) + np.kron(P_y0_x0[1], P_y1_x1[1]))
                P_y0y1_u0_1 = 0.5 * (np.kron(P_y0_x0[1], P_y1_x1[0]) + np.kron(P_y0_x0[0], P_y1_x1[1]))

                P_in = np.array([[P_y0y1_u0_0], [P_y0y1_u0_1]]).squeeze().astype(np.float32)
                border_vectors = self.get_border_vector(P_in.shape[1], self.quantization_level_decoder, num_run=5)

                lut, permutation, p_t_given_x, MI_XY, MI_XT, flag = quantizer.modified_sIB_run(0.5*P_in.transpose(), border_vectors)

                virtual_channel_transition_probs[level, offset:offset+num_lut_per_node] = p_t_given_x

                virtual_channel_llrs[level, offset:offset+num_lut_per_node] = log2_stable(p_t_given_x[0, :]/(p_t_given_x[1, :]+1e-31))

                lut = self.get_LUT(np.array(lut), np.array(permutation), mode='f')
                lut_fs[node_posi] = [lut for _ in range(num_lut_per_node)]

                # build lut for virtual channel u1 -> y0, y1, u0 -> z1
                P_y0y1_u0_0_u1_0 = 0.5 * np.kron(P_y0_x0[0], P_y1_x1[0])
                P_y0y1_u0_1_u1_0 = 0.5 * np.kron(P_y0_x0[1], P_y1_x1[0])
                P_y0y1u0_u1_0 = np.concatenate([P_y0y1_u0_0_u1_0, P_y0y1_u0_1_u1_0])

                P_y0y1_u0_0_u1_1 = 0.5 * np.kron(P_y0_x0[1], P_y1_x1[1])
                P_y0y1_u0_1_u1_1 = 0.5 * np.kron(P_y0_x0[0], P_y1_x1[1])
                P_y0y1u0_u1_1 = np.concatenate([P_y0y1_u0_0_u1_1, P_y0y1_u0_1_u1_1])

                P_in = np.array([[P_y0y1u0_u1_0], [P_y0y1u0_u1_1]]).squeeze().astype(np.float32)
                border_vectors = self.get_border_vector(P_in.shape[1], self.quantization_level_decoder, num_run=5)
                lut, permutation, p_t_given_x, MI_XY, MI_XT, flag = quantizer.modified_sIB_run(0.5*P_in.transpose(), border_vectors)

                virtual_channel_transition_probs[level, offset+stride:offset+stride+num_lut_per_node] = p_t_given_x

                virtual_channel_llrs[level, offset+stride:offset+stride+num_lut_per_node] = log2_stable(p_t_given_x[0,:]/(p_t_given_x[1,:]+1e-31))

                lut = self.get_LUT(np.array(lut), np.array(permutation), mode='g')
                lut_gs[node_posi] = [lut for _ in range(num_lut_per_node)]
            # capacity = np.zeros(self.N)
            # for c in range(self.N):
            #     capacity[c] = mutual_information(0.5*virtual_channel_transition_probs[level, c].transpose())
            # print("level = {:d}, mean_MI = {:3f}".format(level, np.mean(capacity)))

        return lut_fs, lut_gs, virtual_channel_llrs, virtual_channel_transition_probs

