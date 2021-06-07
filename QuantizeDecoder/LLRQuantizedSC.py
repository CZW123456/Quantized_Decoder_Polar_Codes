import numpy as np
from tqdm import tqdm
from quantizers.quantizer.LLROptLSQuantizer import LLRQuantizer

def entropy(x):
    return -np.sum(x * np.log2(x))

class LLRQuantizerSC():

    def __init__(self, N, v):
        self.N = N
        self.v = v

    def f(self, a, b):
        return np.sign(a) * np.sign(b) * np.min([np.abs(a), np.abs(b)])


    def g(self, a, b, u):
        return (1 - 2 * u) * a + b


    def get_llr_density_quanta_after_f(self, llr_density1, llr_density2, llr_quanta1, llr_quanta2):
        llr_quanta = np.zeros(self.v*self.v)
        llr_density = np.zeros(self.v*self.v)
        for i in range(self.v):
            for j in range(self.v):
                f_value = self.f(llr_quanta1[i], llr_quanta2[j])
                llr_density[i*self.v+j] = llr_density1[i] * llr_density2[j]
                llr_quanta[i*self.v+j] = f_value
        return llr_density, llr_quanta

    def get_llr_density_quanta_after_g(self, llr_density1, llr_density2, llr_quanta1, llr_quanta2):
        llr_quanta = np.zeros(2*self.v*self.v)
        llr_density = np.zeros(2*self.v*self.v)
        for u in range(2):
            for i in range(self.v):
                for j in range(self.v):
                    llr_quanta[u*self.v**2+i*self.v+j] = self.g(llr_quanta1[i], llr_quanta2[j], u)
                    llr_density[u*self.v**2+i*self.v+j] = 0.5 * llr_density1[i] * llr_density2[j]
        return llr_density, llr_quanta

    def get_unique_quanta(self, llr_density, llr_quanta):
        unique_quanta = np.unique(llr_quanta)
        unique_density = np.zeros_like(unique_quanta)
        lut = np.zeros_like(llr_density).astype(np.int32)
        for i in range(len(unique_quanta)):
            unique_density[i] = np.sum(llr_density[llr_quanta == unique_quanta[i]])
            lut[llr_quanta == unique_quanta[i]] = i
        return unique_density, unique_quanta, lut

    def get_LUT(self, lut_erasure, lut_merge, mode="f"):
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


    def run(self, channel_llr_density, channel_llr_quanta):

        n = int(np.log2(self.N))
        llr_density = np.zeros((n + 1, self.N, self.v))
        llr_density[0, :, :] = channel_llr_density
        llr_quanta = np.zeros((n + 1, self.N, self.v))
        llr_quanta[0, :, :] = channel_llr_quanta

        lut_fs = {}
        lut_gs = {}

        Quantizer = LLRQuantizer()

        for level in tqdm(range(n)):
            num_bits_per_node = 2 ** (n - level)
            num_node_cur_level = 2 ** level
            stride = num_bits_per_node // 2
            for node in range(num_node_cur_level):
                offset = node * num_bits_per_node
                node_posi = 2**level+node-1

                # Using the llr density of the virtual channel 1 and virtual channel 2 to obtain the llr density after
                # f and g function evaluation
                pllr_density1 = llr_density[level, offset]
                pllr_density2 = llr_density[level, offset + stride]
                pllr_quanta1 = llr_quanta[level, offset]
                pllr_quanta2 = llr_quanta[level, offset + stride]

                # Obtained the evolved LLR density after f and g function
                llr_density_f, llr_quanta_f = self.get_llr_density_quanta_after_f(pllr_density1, pllr_density2, pllr_quanta1, pllr_quanta2)
                llr_density_g, llr_quanta_g = self.get_llr_density_quanta_after_g(pllr_density1, pllr_density2, pllr_quanta1, pllr_quanta2)

                # Merge the symbols with the same quanta first
                unique_llr_density_f, unique_llr_quanta_f, lut_erasure_f = self.get_unique_quanta(llr_density_f, llr_quanta_f)
                unique_llr_density_g, unique_llr_quanta_g, lut_erasure_g = self.get_unique_quanta(llr_density_g, llr_quanta_g)

                # Find the quantizers for compress the alphabet of the evolved LLR density
                llr_compressed_density_f, llr_compressed_quanta_f, lut_merge_f, min_distortion_f = Quantizer.find_OptLS_quantizer(unique_llr_density_f, unique_llr_quanta_f, unique_llr_density_f.shape[0], self.v)
                llr_compressed_density_g, llr_compressed_quanta_g, lut_merge_g, min_distortion_g = Quantizer.find_OptLS_quantizer(unique_llr_density_g, unique_llr_quanta_g, unique_llr_density_g.shape[0], self.v)

                # Get final LUT for f and g function
                lut_f = self.get_LUT(lut_erasure_f, lut_merge_f, "f")
                lut_g = self.get_LUT(lut_erasure_g, lut_merge_g, "g")
                lut_fs[node_posi] = [lut_f for _ in range(num_bits_per_node//2)]
                lut_gs[node_posi] = [lut_g for _ in range(num_bits_per_node//2)]

                # Fill in the llr density and llr quanta so that the LLR density evolution can continuous
                llr_density[level + 1, offset:offset+num_bits_per_node//2] = llr_compressed_density_f
                llr_quanta[level + 1, offset:offset+num_bits_per_node//2] = llr_compressed_quanta_f

                llr_density[level + 1, offset+stride:offset+stride+num_bits_per_node//2] = llr_compressed_density_g
                llr_quanta[level + 1, offset+stride:offset+stride+num_bits_per_node//2] = llr_compressed_quanta_g

        return llr_density, llr_quanta, lut_fs, lut_gs

if __name__ == "__main__":
    q = np.array([-1.7, -0.6, 0.8, 2.1])
    p = np.array([0.25, 0.25, 0.25, 0.25])
    Q = LLRQuantizerSC(512, 128, 4)
    llr_density_f, llr_quanta_f = Q.get_llr_density_quanta_after_f(p, p, q, q)
    llr_density_g, llr_quanta_g = Q.get_llr_density_quanta_after_g(p, p, q, q)
    unique_llr_density_f, unique_llr_quanta_f, lut_erasure_f = Q.get_unique_quanta(llr_density_f, llr_quanta_f)
    unique_llr_density_g, unique_llr_quanta_g, lut_erasure_g = Q.get_unique_quanta(llr_density_g, llr_quanta_g)

    print(unique_llr_quanta_f)
    print(unique_llr_density_f)
    print(unique_llr_quanta_g)
    print(unique_llr_density_g)

