import numpy as np
from tqdm import tqdm
from BitChannelMerge.utils import check_symmetric
from PyIBQuantizer.inf_theory_tools import log2_stable, mutual_information

class DegradeMerger():

    def __init__(self, mu, N, W):
        assert W.shape[0] == 2
        self.mu = mu
        self.v  = mu//2
        self.N = N
        self.W = W

    def LR_sort(self, W):
        L = W.shape[1]
        LLR = np.zeros(L)
        LLR[np.bitwise_and(W[0] == 0, W[1] != 0)] = -np.inf
        LLR[np.bitwise_and(W[0] != 0, W[1] == 0)] = np.inf
        LLR[np.bitwise_and(W[0] != 0, W[1] != 0)] = np.log2(W[0, np.bitwise_and(W[0] != 0, W[1] != 0)]) - np.log2(W[1, np.bitwise_and(W[0] != 0, W[1] != 0)])
        indices = np.argsort(LLR)
        W = W[:, indices]
        return W, indices

    def erasure_symbol_merge(self, W):
        L = W.shape[1]
        cnt = np.sum(W[0, :L//2] == W[1, :L//2])
        lut = np.arange(L)
        if cnt > 0:
            W_erasure = W[:, L//2-cnt:L//2+cnt]
            erasure_probability = np.sum(W_erasure[0])
            W_middle = erasure_probability/2 * np.ones((2,2))
            W_left = W[:, :L//2-cnt]
            W_right = W[:, L//2+cnt:]
            W = np.concatenate([W_left, W_middle, W_right], axis=1)
            lut[L // 2 - cnt:L // 2] = L // 2 - cnt
            lut[L // 2:L // 2 + cnt] = L // 2 - cnt + 1
            lut[L // 2 + cnt:] -= (2 * cnt - 2)
            return W, lut, cnt
        else:
            # W_left = W[:, :L//2]
            # W_right = W[:, L//2:]
            # W_middle =  np.zeros((2, 2))
            # W = np.concatenate([W_left, W_middle, W_right], axis=1)
            return W, lut, cnt

    def capacity(self, a, b):
        if a != 0 and b != 0:
            z = -(a+b) * np.log2((a+b)/2) + a * np.log2(a) + b * np.log2(b)
        elif a == 0 and b != 0:
            z = b
        elif a != 0 and b == 0:
            z = a
        else:
            z = 0
        return z

    def degrading_merge(self, W):
        L_origin = W.shape[1]
        L = W.shape[1]
        lut = np.arange(L)
        if L <= self.mu:
            return W, lut
        else:
            while L > self.mu:
                min_deltaI = np.inf
                min_index = -1
                # find two consecutive symbols so that merging these two symbols has minimum capacity loss
                for i in range(L//2 - 1):
                    a1 = W[0, i]
                    b1 = W[1, i]
                    a2 = W[0, i+1]
                    b2 = W[1, i+1]
                    deltaI = self.capacity(a1, b1) + self.capacity(a2, b2) - self.capacity(a1+a2, b1+b2)
                    # print("a1 = {:4e} b1 = {:4e} a2 = {:4e} b2 = {:4e}".format(a1, b1, a2, b2))
                    if deltaI < min_deltaI:
                        min_deltaI = deltaI
                        min_index = i
                # merge two symbols to a new symbol
                # print("min_index = {:d}".format(min_index))
                W[0, min_index] = W[0, min_index] + W[0, min_index+1]
                W[1, min_index] = W[1, min_index] + W[1, min_index+1]
                W[0, L-min_index-1] = W[0, L-min_index-2] + W[0, L-min_index-1]
                W[1, L-min_index-1] = W[1, L-min_index-2] + W[1, L-min_index-1]
                W = np.delete(W, min_index+1, axis=1)
                W = np.delete(W, L-min_index-3, axis=1)
                # LUT update
                lut[min_index] = lut[min_index + 1]
                lut = np.delete(lut, min_index + 1)
                lut = np.delete(lut, L - min_index - 3)
                L = W.shape[1]

            # translate lut
            Az = np.zeros(self.mu + 1)
            Az[1:] = lut + 1
            result_lut = np.zeros(L_origin)
            for i in range(lut.shape[0]):
                begin = int(Az[i])
                end = int(Az[i + 1])
                result_lut[begin:end] = i
            return W, result_lut

    def get_lut(self, lut_erasure, lut_merge, permutation, mode="f"):
        L = len(lut_erasure)
        lut_erasure = np.arange(L)
        if lut_erasure[-1] != L - 1:
            erasure_begin = (lut_erasure[-1] - 1) // 2
            cnt = L//2 - erasure_begin
            erasure_end = L//2 + cnt
            lut_merge_complete = np.zeros_like(lut_erasure)
            lut_merge_complete[:erasure_begin] = lut_merge[:erasure_begin]
            lut_merge_complete[erasure_begin:L//2] = self.mu // 2 - 1
            lut_merge_complete[L//2:erasure_end] = self.mu // 2
            lut_merge_complete[erasure_end:] = lut_merge[erasure_begin+2:]
        else:
            lut_merge_complete = lut_merge

        if mode == 'f':
            LUT = np.zeros((self.mu, self.mu)).astype(np.int32)

            for i in range(self.mu):
                indices = np.where(lut_merge_complete == i)[0]
                symbols = permutation[indices]
                y = symbols // self.mu
                x = symbols %  self.mu
                LUT[y, x] = i
            return LUT
        elif mode == 'g':
            LUT = np.zeros((2, self.mu, self.mu))
            for i in range(self.mu):
                indices = np.where(lut_merge_complete == i)[0]
                symbols = permutation[indices]
                u0 = symbols // (self.mu*self.mu)
                tmp = symbols - u0*self.mu*self.mu
                y = tmp // self.mu
                x = tmp %  self.mu
                LUT[u0, y, x] = i
            return LUT

    def bit_channel_degrading_run(self):
        n = int(np.log2(self.N))
        virtual_channel_transition_probs = np.zeros((n, self.N, 2, self.mu))
        virtual_channel_llrs = np.zeros((n, self.N, self.mu))
        W_AWGN = np.expand_dims(self.W, axis=0).repeat(self.N, axis=0)
        lut_f = {}
        lut_g = {}
        for level in tqdm(range(n)):
            num_bits_per_node = 2 ** (n - level)
            num_node_cur_level = 2 ** level
            num_lut_per_node = num_bits_per_node // 2
            if level == 0:
                probs = W_AWGN
            else:
                probs = virtual_channel_transition_probs[level - 1]
            for node in range(num_node_cur_level):

                offset = num_bits_per_node * node
                stride = num_bits_per_node // 2
                node_posi = 2 ** level + node - 1

                lut_f[node_posi] = []
                lut_g[node_posi] = []

                P_y0_x0 = probs[offset]
                P_y1_x1 = probs[offset + stride]

                assert np.all(P_y0_x0 == P_y1_x1)

                # # Quantize virtual channel 1
                P_y0y1_u0_0 = 0.5 * (np.kron(P_y0_x0[0], P_y1_x1[0]) + np.kron(P_y0_x0[1], P_y1_x1[1]))
                P_y0y1_u0_1 = 0.5 * (np.kron(P_y0_x0[1], P_y1_x1[0]) + np.kron(P_y0_x0[0], P_y1_x1[1]))

                if node_posi == 0:
                    print()

                P_up = np.array([[P_y0y1_u0_0], [P_y0y1_u0_1]]).squeeze().astype(np.float64)
                P_up, permutation_indices = self.LR_sort(P_up)

                P_up_after_erasure, lut_erasure, cnt = self.erasure_symbol_merge(P_up)
                P_up_after_merge, lut_merge  = self.degrading_merge(P_up)

                LUT = self.get_lut(lut_erasure, lut_merge, permutation_indices, mode='f')
                lut_f[node_posi] = [LUT for _ in range(num_lut_per_node)]
                virtual_channel_transition_probs[level, offset:offset+num_lut_per_node] = P_up_after_merge
                virtual_channel_llrs[level, offset:offset+num_lut_per_node] = log2_stable(P_up_after_merge[0] / (P_up_after_merge[1]))

                # Quantize virtual channel 2
                P_y0y1_u0_0_u1_0 = 0.5 * np.kron(P_y0_x0[0], P_y1_x1[0])
                P_y0y1_u0_1_u1_0 = 0.5 * np.kron(P_y0_x0[1], P_y1_x1[0])
                P_y0y1u0_u1_0 = np.concatenate([P_y0y1_u0_0_u1_0, P_y0y1_u0_1_u1_0])

                P_y0y1_u0_0_u1_1 = 0.5 * np.kron(P_y0_x0[1], P_y1_x1[1])
                P_y0y1_u0_1_u1_1 = 0.5 * np.kron(P_y0_x0[0], P_y1_x1[1])
                P_y0y1u0_u1_1 = np.concatenate([P_y0y1_u0_0_u1_1, P_y0y1_u0_1_u1_1])

                P_down = np.array([[P_y0y1u0_u1_0], [P_y0y1u0_u1_1]]).squeeze().astype(np.float64)

                P_down, permutation_indices = self.LR_sort(P_down)

                P_down_after_erasure, lut_erasure, cnt = self.erasure_symbol_merge(P_down)
                P_down_after_merge, lut_merge = self.degrading_merge(P_down)

                LUT = self.get_lut(lut_erasure, lut_merge, permutation_indices, mode='g')
                lut_g[node_posi] = [LUT for _ in range(num_lut_per_node)]
                virtual_channel_transition_probs[level, offset + stride:offset + stride + num_lut_per_node] = P_down_after_merge
                virtual_channel_llrs[level, offset + stride:offset + stride + num_lut_per_node] = log2_stable(P_down_after_merge[0] / (P_down_after_merge[1]))

        Pe = np.zeros(self.N)
        for i in range(self.N):
            Pe[i] = 0.5 * np.sum(np.min(virtual_channel_transition_probs[-1, i, :], axis=0))
        return virtual_channel_llrs, Pe, lut_f, lut_g
