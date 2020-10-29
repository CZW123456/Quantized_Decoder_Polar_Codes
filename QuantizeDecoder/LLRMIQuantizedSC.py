import numpy as np
from tqdm import tqdm
import QuantizeDecoder
from quantizers.quantizer.LLRMIQuantizer import LLRMIQuantizer
from PyIBQuantizer.inf_theory_tools import mutual_information

class LLRMIQuantizerSC():


    def __init__(self, N, K, v):
        self.N = N
        self.K = K
        self.v = v


    def f(self, a, b):
        return np.log((1 + np.exp(a + b))/(np.exp(a) + np.exp(b)))
        # return np.sign(a) * np.sign(b) * np.min([np.abs(a), np.abs(b)])


    def g(self, a, b, u):
        return (1 - 2 * u) * a + b


    def get_llr_density_quanta_after_f(self, llr_density1, llr_density2, llr_quanta1, llr_quanta2):
        llr_quanta = np.zeros(self.v*self.v)
        llr_density = np.zeros((2, self.v*self.v))
        for i in range(self.v):
            for j in range(self.v):
                f_value = self.f(llr_quanta1[i], llr_quanta2[j])
                llr_density[0, i * self.v + j] = (llr_density1[0, i] * llr_density2[0, j] + llr_density1[1, i] * llr_density2[1, j]) / 2
                llr_density[1, i * self.v + j] = (llr_density1[0, i] * llr_density2[1, j] + llr_density1[1, i] * llr_density2[0, j]) / 2
                llr_quanta[i*self.v+j] = f_value
        return llr_density, llr_quanta


    def get_llr_density_quanta_after_g(self, llr_density1, llr_density2, llr_quanta1, llr_quanta2):
        llr_quanta = np.zeros(2*self.v*self.v)
        llr_density = np.zeros((2, 2*self.v*self.v))
        # u1 = 0
        for i in range(self.v):
            for j in range(self.v):
                llr_density[0, i * self.v + j] = 0.5 * llr_density1[0, i] * llr_density2[0, j]
                llr_density[1, i * self.v + j] = 0.5 * llr_density1[1, i] * llr_density2[1, j]
                llr_quanta[i*self.v + j] = self.g(llr_quanta1[i], llr_quanta2[j], 0)
        # u1 = 1
        for i in range(self.v):
            for j in range(self.v):
                llr_density[0, self.v ** 2 + i * self.v + j] = 0.5 * llr_density1[1, i] * llr_density2[0, j]
                llr_density[1, self.v ** 2 + i * self.v + j] = 0.5 * llr_density1[0, i] * llr_density2[1, j]
                llr_quanta[self.v**2 + i*self.v + j] = self.g(llr_quanta1[i], llr_quanta2[j], 1)
        return llr_density, llr_quanta


    def get_unique_quanta(self, llr_density, llr_quanta):
        unique_quanta = np.unique(llr_quanta)
        unique_density = np.zeros((2, unique_quanta.shape[0]))
        lut = np.zeros_like(llr_quanta).astype(np.int32)
        for i in range(len(unique_quanta)):
            unique_density[:, i] = np.sum(llr_density[:, llr_quanta == unique_quanta[i]], axis=1).transpose()
            lut[llr_quanta == unique_quanta[i]] = i
        return unique_density, unique_quanta, lut


    def get_LUT_with_erasure(self, lut_erasure, lut_merge, mode="f"):
        lut_merge = lut_merge.squeeze()
        lut_erasure = lut_erasure.squeeze()
        if mode == "f":
            lut = np.zeros((self.v, self.v)).astype(np.int32)
            for i in range(self.v):
                for j in range(self.v):
                    symbol_erasure = lut_erasure[i*self.v+j]
                    symbol_merge = lut_merge[symbol_erasure]
                    lut[i, j] = symbol_merge
            return lut
        else:
            lut = np.zeros((2, self.v, self.v)).astype(np.int32)
            for u in range(2):
                for i in range(self.v):
                    for j in range(self.v):
                        symbol_erasure = lut_erasure[u*self.v*self.v + i*self.v + j]
                        symbol_merge = lut_merge[symbol_erasure]
                        lut[u, i, j] = symbol_merge
            return lut

    def get_LUT(self, lut_merge, mode="f"):
        lut_merge = lut_merge.squeeze()
        if mode == "f":
            lut = np.zeros((self.v, self.v)).astype(np.int32)
            for i in range(self.v):
                for j in range(self.v):
                    symbol_merge = lut_merge[i*self.v+j]
                    lut[i, j] = symbol_merge
            return lut
        else:
            lut = np.zeros((2, self.v, self.v)).astype(np.int32)
            for u in range(2):
                for i in range(self.v):
                    for j in range(self.v):
                        symbol_merge = lut_merge[u*self.v*self.v + i*self.v + j]
                        lut[u, i, j] = symbol_merge
            return lut

    def generate_LUTs(self, channel_llr_density, channel_llr_quanta):

        n = int(np.log2(self.N))
        llr_density = np.zeros((n + 1, self.N, 2, self.v))
        llr_density[0, :, :, :] = channel_llr_density
        llr_quanta = np.zeros((n + 1, self.N, self.v))
        llr_quanta[0, :, :] = channel_llr_quanta

        lut_fs = {}
        lut_gs = {}

        Quantizer_py = QuantizeDecoder.LLRMIQuantizer.LLRMIQuantizer(self.K)
        Quantizer = LLRMIQuantizer()

        for level in tqdm(range(n)):
            num_bits_per_node = 2 ** (n - level)
            num_node_cur_level = 2 ** level
            stride = num_bits_per_node // 2
            mean_I = 0
            for node in range(num_node_cur_level):
                offset = node * num_bits_per_node
                node_posi = 2**level+node-1

                # Using the llr density of the virtual channel 1 and virtual channel 2 to obtain the llr density after
                # f and g function evaluation
                pllr_density1 = llr_density[level, offset]
                pllr_density2 = llr_density[level, offset + stride]
                pllr_quanta1 = llr_quanta[level, offset]
                pllr_quanta2 = llr_quanta[level, offset + stride]

                I_2 = 2 * mutual_information(0.5 * pllr_density1.transpose())

                # Obtained the evolved LLR density after f and g function
                llr_density_f, llr_quanta_f = self.get_llr_density_quanta_after_f(pllr_density1, pllr_density2, pllr_quanta1, pllr_quanta2)
                llr_density_g, llr_quanta_g = self.get_llr_density_quanta_after_g(pllr_density1, pllr_density2, pllr_quanta1, pllr_quanta2)

                # Merge the symbols with the same quanta first
                # unique_llr_density_f, unique_llr_quanta_f, lut_erasure_f = self.get_unique_quanta(llr_density_f, llr_quanta_f)
                unique_llr_density_g, unique_llr_quanta_g, lut_erasure_g = self.get_unique_quanta(llr_density_g, llr_quanta_g)

                # Find the quantizers for compress the alphabet of the evolved LLR density
                I_f = Quantizer_py.compute_mutual_information_discrete(llr_density_f,
                                                                       llr_quanta_f)
                I_g = Quantizer_py.compute_mutual_information_discrete(llr_density_g,
                                                                       llr_quanta_g)

                permutation = np.argsort(np.log(llr_density_f[0]/llr_density_f[1]))

                # plx_compressed_f_py, llr_quanta_f_py, lut_merge_f_py, I_f_quantized_py, plx_py, table_py = Quantizer_py.find_quantizer_decoder(
                #     unique_llr_density_f,
                #     unique_llr_quanta_f,
                #     unique_llr_quanta_f.shape[0],
                #     self.K)
                if llr_quanta_f.shape[0] <= self.K:
                    print()

                plx_compressed_f, llr_quanta_f, lut_merge_f, I_f_quantized, plx, table = Quantizer.find_quantizer_decoder(llr_density_f,
                                                                                                                          permutation,
                                                                                                                          llr_quanta_f.shape[0],
                                                                                                                          self.K)
                # print(np.all(plx == plx_py))
                # print(np.all(table == table_py))
                # tmp1 = np.abs(table_py - table)

                I_f_quantized = Quantizer_py.compute_mutual_information_discrete(plx_compressed_f,
                                                                                 llr_quanta_f)

                permutation = np.argsort(np.log(llr_density_g[0] / llr_density_g[1]))
                # plx_compressed_g_py, llr_quanta_g_py, lut_merge_g_py, I_g_quantized_py, plx_py, table_py = Quantizer_py.find_quantizer_decoder(
                #     unique_llr_density_g,
                #     unique_llr_quanta_g,
                #     unique_llr_quanta_g.shape[0],
                #     self.K)
                if llr_quanta_g.shape[0] <= self.K:
                    print()
                plx_compressed_g, llr_quanta_g, lut_merge_g, I_g_quantized, plx, table = Quantizer.find_quantizer_decoder(llr_density_g,
                                                                                                                          permutation,
                                                                                                                          llr_quanta_g.shape[0],
                                                                                                                          self.K)
                # print(np.all(plx == plx_py))
                # print(np.all(table == table_py))

                I_g_quantized = Quantizer_py.compute_mutual_information_discrete(plx_compressed_g,
                                                                                 llr_quanta_g)
                # print(np.all(lut_merge_f_py == lut_merge_f))
                # print(np.all(plx_compressed_f == plx_compressed_f_py))
                # print(np.all(llr_quanta_f == llr_quanta_f_py))
                # print(I_f_quantized == I_f_quantized_py)
                # print(np.all(lut_merge_g_py == lut_merge_g))
                # print(np.all(plx_compressed_g == plx_compressed_g_py))
                # print(np.all(llr_quanta_g == llr_quanta_g_py))
                # print(I_g_quantized == I_g_quantized_py)



                mean_I += (I_g_quantized + I_f_quantized)
                # if I_g < I_g_quantized or I_f < I_f_quantized:
                #     print("overflow")


                # Get final LUT for f and g function
                lut_f = self.get_LUT(lut_merge_f, "f")
                lut_g = self.get_LUT(lut_merge_g, "g")
                lut_fs[node_posi] = [lut_f for _ in range(num_bits_per_node//2)]
                lut_gs[node_posi] = [lut_g for _ in range(num_bits_per_node//2)]

                # Fill in the llr density and llr quanta so that the LLR density evolution can continuous
                llr_density[level + 1, offset:offset+num_bits_per_node//2] = plx_compressed_f
                llr_quanta[level + 1, offset:offset+num_bits_per_node//2] = llr_quanta_f

                llr_density[level + 1, offset+stride:offset+stride+num_bits_per_node//2] = plx_compressed_g
                llr_quanta[level + 1, offset+stride:offset+stride+num_bits_per_node//2] = llr_quanta_g
            print("level = {:d}, mean capacity = {:f}".format(level, mean_I/num_node_cur_level/2))

        return llr_density, llr_quanta, lut_fs, lut_gs


