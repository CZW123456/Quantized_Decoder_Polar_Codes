import numpy as np
from tqdm import tqdm
from BitChannelMerge.utils import get_AWGN_transition_probability_upgrading
import pickle as pkl
import os
from PyIBQuantizer.inf_theory_tools import log2_stable
import argparse

class UpgradeMerger():

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
        LLR[np.bitwise_and(W[0] != 0, W[1] != 0)] = np.log(W[0, np.bitwise_and(W[0] != 0, W[1] != 0)]) - np.log(W[1, np.bitwise_and(W[0] != 0, W[1] != 0)])
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
            lut[L//2-cnt:L//2] = L//2-cnt
            lut[L//2:L//2+cnt] = L//2-cnt+1
            lut[L//2+cnt:] -= (2*cnt - 2)
            return W, lut
        else:
            return W, lut


    def delta_capacity_basic(self, a, b):
        if a != 0 and b != 0:
            z = -(a+b) * np.log2((a+b)/2) + a * np.log2(a) + b * np.log2(b)
        elif a == 0 and b != 0:
            z = b
        elif a != 0 and b == 0:
            z = a
        else:
            z = 0
        return z


    def delta_capacity_lemma9(self, a1, a2, b1, b2):
        I1 = -self.delta_capacity_basic(a1, b1)
        I2 = -self.delta_capacity_basic(a2, b2)

        if a2/b2 < np.inf:
            lambda2 = a2/b2
            alpha2 = lambda2 * (a1 + b1)/(lambda2+1)
            beta2 = (a1 + b1)/(lambda2+1)
        else:
            alpha2 = a1 + b1
            beta2 = 0

        I3 = self.delta_capacity_basic(a2 + alpha2, b2 + beta2)

        z = I1 + I2 + I3
        return z

    def delta_capacity_lemma11(self, a1, a2, a3, b1, b2, b3):
        I1 = -self.delta_capacity_basic(a1, b1)
        I2 = -self.delta_capacity_basic(a2, b2)
        I3 = -self.delta_capacity_basic(a3, b3)

        lambda1 = a1 / b1

        if a3/b3 < np.inf:
            lambda3 = a3/b3
            alpha1 = lambda1 * (lambda3 * b2 - a2) / (lambda3 - lambda1)
            beta1 = (lambda3 * b2 - a2) / (lambda3 - lambda1)
            alpha3 = lambda3 * (a2 - lambda1 * b2) / (lambda3 - lambda1)
            beta3 = (a2 - lambda1 * b2) / (lambda3 - lambda1)
        else:
            alpha1 = lambda1 * b2
            beta1 = b2
            alpha3 = a2 - lambda1 * b2
            beta3 = 0

        I4 = self.delta_capacity_basic(a3 + alpha3, b3 + beta3)
        I5 = self.delta_capacity_basic(a1 + alpha1, b1 + beta1)

        z = I1 + I2 + I3 + I4 + I5

        return z


    def upgrading_merge_lemma9_plus_lemma11(self, W):
        L = W.shape[1]
        if L <= self.mu:
            return W
        else:
            eps = 1e-3
            while L > self.mu:
                W_first_half = W[:, :L//2]
                LR = W_first_half[0]/(W_first_half[1])
                # check whether there are two consecutive symbols which have extremely similar LLR
                numerical_warining = False
                for i in range(L//2 - 1):
                    ratio = LR[i]/LR[i+1]
                    if ratio < 1 + eps:
                        numerical_warining = True
                        break
                if numerical_warining:
                    # apply lemma9
                    min_deltaI = 1e308
                    min_index = -1
                    for i in range(L//2-1):
                        a2 = W[0, i]
                        b2 = W[1, i]
                        a1 = W[0, i+1]
                        b1 = W[1, i+1]

                        deltaI = self.delta_capacity_lemma9(a1, a2, b1, b2)

                        if deltaI < min_deltaI:
                            min_deltaI = deltaI
                            min_index = i

                    if min_index == -1:
                        for k in range(L//2):
                            if np.sum(W[:, k]) < 1e-20:
                                min_index = k
                                W = np.delete(W[:, min_index])
                                W = np.delete(W[:, L-min_index-2])
                                L = W.shape[1]
                                break
                        continue

                    a2 = W[0, min_index]
                    b2 = W[1, min_index]
                    a1 = W[0, min_index+1]
                    b1 = W[1, min_index+1]

                    if a2/b2 < np.inf:
                        lambda2 = a2/b2
                        alpha2 = lambda2 * (a1 + b1)/(lambda2 + 1)
                        beta2 = (a1 + b1) / (lambda2 + 1)
                    else:
                        alpha2 = a1 + b1
                        beta2 = 0

                    W[0, min_index] = a2 + alpha2
                    W[1, min_index] = b2 + beta2
                    W[0, L-min_index-1] = b2 + beta2
                    W[1, L-min_index-1] = a2 + alpha2
                    W = np.delete(W, min_index + 1, axis=1)
                    W = np.delete(W, L - min_index - 3, axis=1)
                    L = W.shape[1]

                else:
                    min_deltaI = 1e308
                    min_index = -1

                    for i in range(L//2 - 2):
                        a3 = W[0, i]
                        b3 = W[1, i]
                        a2 = W[0, i + 1]
                        b2 = W[1, i + 1]
                        a1 = W[0, i + 2]
                        b1 = W[1, i + 2]

                        deltaI = self.delta_capacity_lemma11(a1, a2, a3, b1, b2, b3)

                        if deltaI < min_deltaI:
                            min_deltaI = deltaI
                            min_index = i

                    if min_index == -1:
                        for k in range(L//2):
                            if np.sum(W[:, k]) < 1e-20:
                                min_index = k
                                W = np.delete(W[:, min_index])
                                W = np.delete(W[:, L-min_index-2])
                                L = W.shape[1]
                                break
                        continue

                    a3 = W[0, min_index]
                    b3 = W[1, min_index]
                    a2 = W[0, min_index+1]
                    b2 = W[1, min_index+1]
                    a1 = W[0, min_index+2]
                    b1 = W[1, min_index+2]

                    lambda1 = a1 / b1

                    if a3 / b3 < np.inf:
                        lambda3 = a3 / b3
                        alpha1 = lambda1 * (lambda3 * b2 - a2) / (lambda3 - lambda1)
                        beta1 = (lambda3 * b2 - a2) / (lambda3 - lambda1)
                        alpha3 = lambda3 * (a2 - lambda1 * b2) / (lambda3 - lambda1)
                        beta3 = (a2 - lambda1 * b2) / (lambda3 - lambda1)
                    else:
                        alpha1 = lambda1 * b2
                        beta1 = b2
                        alpha3 = a2 - lambda1 * b2
                        beta3 = 0

                    W[0, min_index] = a3 + alpha3
                    W[1, min_index] = b3 + beta3

                    W[0, min_index + 1] = a1 + alpha1
                    W[1, min_index + 1] = b1 + beta1

                    W[0, L - min_index - 1] = b3 + beta3
                    W[1, L - min_index - 1] = a3 + alpha3

                    W[0, L - min_index - 2] = b1 + beta1
                    W[1, L - min_index - 2] = a1 + alpha1

                    W = np.delete(W, min_index + 2, axis=1)
                    W = np.delete(W, L - min_index - 4, axis=1)

                    L = W.shape[1]
            return W

    def upgrading_merge_lemma9(self, W):
        L_origin = W.shape[1]
        L = W.shape[1]
        lut = np.arange(L)
        if L <= self.mu:
            return W, lut
        else:
            while L > self.mu:
                # apply lemma9
                min_deltaI = 1e308
                min_index = -1
                for i in range(L//2-1):
                    a2 = W[0, i]
                    b2 = W[1, i]
                    a1 = W[0, i+1]
                    b1 = W[1, i+1]

                    deltaI = self.delta_capacity_lemma9(a1, a2, b1, b2)

                    if deltaI < min_deltaI:
                        min_deltaI = deltaI
                        min_index = i

                if min_index == -1:
                    for k in range(L//2):
                        if np.sum(W[:, k]) < 1e-20:
                            min_index = k
                            W = np.delete(W[:, min_index])
                            W = np.delete(W[:, L-min_index-2])
                            L = W.shape[1]
                            break
                    continue

                a2 = W[0, min_index]
                b2 = W[1, min_index]
                a1 = W[0, min_index+1]
                b1 = W[1, min_index+1]

                if a2/b2 < np.inf:
                    lambda2 = a2/b2
                    alpha2 = lambda2 * (a1 + b1)/(lambda2 + 1)
                    beta2 = (a1 + b1) / (lambda2 + 1)
                else:
                    alpha2 = a1 + b1
                    beta2 = 0

                W[0, min_index] = a2 + alpha2
                W[1, min_index] = b2 + beta2
                W[0, L-min_index-1] = b2 + beta2
                W[1, L-min_index-1] = a2 + alpha2
                W = np.delete(W, min_index + 1, axis=1)
                W = np.delete(W, L - min_index - 3, axis=1)
                # LUT update
                lut[min_index] = lut[min_index + 1]
                lut = np.delete(lut, min_index+1)
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
            cnt = L // 2 - erasure_begin
            erasure_end = L // 2 + cnt
            lut_merge_complete = np.zeros_like(lut_erasure)
            lut_merge_complete[:erasure_begin] = lut_merge[:erasure_begin]
            lut_merge_complete[erasure_begin:L // 2] = self.mu // 2 - 1
            lut_merge_complete[L // 2:erasure_end] = self.mu // 2
            lut_merge_complete[erasure_end:] = lut_merge[erasure_begin + 2:]
        else:
            lut_merge_complete = lut_merge

        if mode == 'f':
            LUT = np.zeros((self.mu, self.mu)).astype(np.int32)
            for i in range(self.mu):
                indices = np.where(lut_merge_complete == i)[0]
                symbols = permutation[indices]
                y = symbols // self.mu
                x = symbols % self.mu
                LUT[y, x] = i
            return LUT
        elif mode == 'g':
            LUT = np.zeros((2, self.mu, self.mu))
            for i in range(self.mu):
                indices = np.where(lut_merge_complete == i)[0]
                symbols = permutation[indices]
                u0 = symbols // (self.mu * self.mu)
                tmp = symbols - u0 * self.mu * self.mu
                y = tmp // self.mu
                x = tmp % self.mu
                LUT[u0, y, x] = i
            return LUT

    def bit_channel_upgrading_run(self):
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
                node_posi = 2**level+node-1

                lut_f[node_posi] = []
                lut_g[node_posi] = []

                P_y0_x0 = probs[offset]
                P_y1_x1 = probs[offset + stride]

                # build lut for virtual channel u0 -> y0, y1 -> z0
                P_y0y1_u0_0 = 0.5 * (np.kron(P_y0_x0[0], P_y1_x1[0]) + np.kron(P_y0_x0[1], P_y1_x1[1]))
                P_y0y1_u0_1 = 0.5 * (np.kron(P_y0_x0[1], P_y1_x1[0]) + np.kron(P_y0_x0[0], P_y1_x1[1]))

                P_up = np.array([[P_y0y1_u0_0], [P_y0y1_u0_1]]).squeeze().astype(np.float64)
                P_up, permutation_indices = self.LR_sort(P_up) # sort symbols according to LR (LLR) in ascending order
                P_up_after_erasure, lut_erasure = self.erasure_symbol_merge(P_up)
                P_up_after_merge, lut_merge = self.upgrading_merge_lemma9(P_up)
                LUT = self.get_lut(lut_erasure, lut_merge, permutation_indices, mode='f')
                lut_f[node_posi] = [LUT for _ in range(num_lut_per_node)]
                virtual_channel_transition_probs[level, offset:offset + num_lut_per_node] = P_up_after_merge
                virtual_channel_llrs[level, offset:offset + num_lut_per_node] = log2_stable(P_up_after_merge[0] / (P_up_after_merge[1]))

                # build lut for virtual channel u1 -> y0, y1, u0 -> z1
                P_y0y1_u0_0_u1_0 = 0.5 * np.kron(P_y0_x0[0], P_y1_x1[0])
                P_y0y1_u0_1_u1_0 = 0.5 * np.kron(P_y0_x0[1], P_y1_x1[0])
                P_y0y1u0_u1_0 = np.concatenate([P_y0y1_u0_0_u1_0, P_y0y1_u0_1_u1_0])

                P_y0y1_u0_0_u1_1 = 0.5 * np.kron(P_y0_x0[1], P_y1_x1[1])
                P_y0y1_u0_1_u1_1 = 0.5 * np.kron(P_y0_x0[0], P_y1_x1[1])
                P_y0y1u0_u1_1 = np.concatenate([P_y0y1_u0_0_u1_1, P_y0y1_u0_1_u1_1])

                P_down = np.array([[P_y0y1u0_u1_0], [P_y0y1u0_u1_1]]).squeeze().astype(np.float64)
                P_down, permutation_indices = self.LR_sort(P_down)
                P_down_after_erasure, lut_erasure = self.erasure_symbol_merge(P_down)
                P_down_after_merge, lut_merge = self.upgrading_merge_lemma9(P_down)
                LUT = self.get_lut(lut_erasure, lut_merge, permutation_indices, mode='g')
                lut_g[node_posi] = [LUT for _ in range(num_lut_per_node)]
                virtual_channel_transition_probs[level, offset + stride:offset + stride + num_lut_per_node] = P_down_after_merge
                virtual_channel_llrs[level, offset + stride:offset + stride + num_lut_per_node] = log2_stable(P_down_after_merge[0]/(P_down_after_merge[1] + 1e-31))

        Pe = np.zeros(self.N)
        for i in range(self.N):
            Pe[i] = 0.5 * np.sum(np.min(virtual_channel_transition_probs[-1, i, :], axis=0))

        return virtual_channel_llrs, Pe, lut_f, lut_g



