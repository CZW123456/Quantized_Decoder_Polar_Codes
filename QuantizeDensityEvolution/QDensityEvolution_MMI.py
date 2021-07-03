import numpy as np
from quantizers.quantizer.MMI import MMIQuantizer
from tqdm import tqdm
from PyIBQuantizer.inf_theory_tools import log2_stable, mutual_information


class QDensityEvolutionMMI():

    def __init__(self, N, quantization_level_decoder):
        self.N = N
        self.quantization_level_decoder = quantization_level_decoder

    def get_lut_from_Q(self, Q, num_y0, num_y1, mode='f'):
        if mode == 'f':
            lut = np.zeros((num_y0, num_y1))
            cnt = 0
            for i in range(num_y0):
                for j in range(num_y1):
                    lut[i, j] = np.where(Q[:, cnt] == 1)[0]
                    cnt += 1
        elif mode == 'g':
            lut = np.zeros((2, num_y0, num_y1))
            cnt = 0
            for i in range(num_y0):
                for j in range(num_y1):
                    for n in range(2):
                        lut[n, i, j] = np.where(Q[:, cnt] == 1)[0]
                        cnt += 1
        else:
            raise NotImplementedError
        return lut


    def run(self, channel_symbol_probs):
        n = int(np.log2(self.N))

        virtual_channel_transition_probs = np.zeros((n, self.N, 2, self.quantization_level_decoder))
        virtual_channel_llrs = np.zeros((n, self.N, self.quantization_level_decoder))

        channel_symbol_probs = np.expand_dims(channel_symbol_probs, axis=0).repeat(self.N, axis=0)

        quantize_level_channel = channel_symbol_probs.shape[2]

        quantizer = MMIQuantizer(px1=0.5, px_minus1=0.5)
        lut_fs = {}
        lut_gs = {}

        for level in tqdm(range(n)):

            num_bits_per_node = 2**(n-level)
            num_lut_per_node = num_bits_per_node // 2
            num_node_cur_level = 2**level

            if level == 0:
                probs = channel_symbol_probs
                n_level_y0 = quantize_level_channel
                n_level_y1 = quantize_level_channel
            else:
                probs = virtual_channel_transition_probs[level - 1]
                n_level_y0 = self.quantization_level_decoder
                n_level_y1 = self.quantization_level_decoder

            mean_I = 0

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

                Q, P_z0_u0, _, _ = quantizer.find_opt_quantizer(P_in, self.quantization_level_decoder)

                I_f_quantized = mutual_information(0.5 * P_z0_u0.transpose())

                virtual_channel_transition_probs[level, offset:offset+num_lut_per_node] = P_z0_u0

                virtual_channel_llrs[level, offset:offset+num_lut_per_node] = log2_stable(P_z0_u0[0]/(P_z0_u0[1]+1e-31))

                lut = self.get_lut_from_Q(Q, n_level_y0, n_level_y1, mode='f')
                lut_fs[node_posi] = [lut for _ in range(num_lut_per_node)]

                # build lut for virtual channel u1 -> y0, y1, u0 -> z1
                P_y0y1_u0_0_u1_0 = 0.5 * np.kron(P_y0_x0[0], P_y1_x1[0])
                P_y0y1_u0_1_u1_0 = 0.5 * np.kron(P_y0_x0[1], P_y1_x1[0])
                tmp = np.dstack((P_y0y1_u0_0_u1_0, P_y0y1_u0_1_u1_0)).squeeze()
                P_y0y1u0_u1_0 = np.reshape(tmp, [2 * tmp.shape[0]])
                P_y0y1_u0_0_u1_1 = 0.5 * np.kron(P_y0_x0[1], P_y1_x1[1])
                P_y0y1_u0_1_u1_1 = 0.5 * np.kron(P_y0_x0[0], P_y1_x1[1])
                tmp = np.dstack((P_y0y1_u0_0_u1_1, P_y0y1_u0_1_u1_1)).squeeze()
                P_y0y1u0_u1_1 = np.reshape(tmp, [2 * tmp.shape[0]])

                P_in = np.array([[P_y0y1u0_u1_0], [P_y0y1u0_u1_1]]).squeeze().astype(np.float32)

                Q, P_z1_u1, Az, indices = quantizer.find_opt_quantizer(P_in, self.quantization_level_decoder)

                I_g_quantized = mutual_information(0.5 * P_z1_u1.transpose())

                virtual_channel_transition_probs[level, offset+stride:offset+stride+num_lut_per_node] = P_z1_u1


                virtual_channel_llrs[level, offset+stride:offset+stride+num_lut_per_node] = log2_stable(P_z1_u1[0]/(P_z1_u1[1]+1e-31))

                lut = self.get_lut_from_Q(Q, n_level_y0, n_level_y1, mode='g')
                lut_gs[node_posi] = [lut for _ in range(num_lut_per_node)]
                mean_I += (I_f_quantized + I_g_quantized)

            # print("level = {:d}, mean capacity = {:f}".format(level, mean_I / num_node_cur_level / 2))

        return lut_fs, lut_gs, virtual_channel_llrs, virtual_channel_transition_probs







