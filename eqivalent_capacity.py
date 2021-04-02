import numpy as np
from PyIBQuantizer.inf_theory_tools import mutual_information
import argparse
from PolarCodesUtils.CodeConstruction import PolarCodeConstructor
from PolarCodesUtils.IdentifyNodes import NodeIdentifier
from QuantizeDecoder.QDensityEvolution_MMI import channel_transition_probability_table
from quantizers.quantizer.MMI import MMIQuantizer
import os
import pickle as pkl

class EqivalentCapacity():

    def __init__(self, N, K, frozen_bits, message_bits, node_type, virtual_channel_probs):
        self.N = N
        self.K = K
        self.frozen_bits = frozen_bits
        self.msg_bits = message_bits
        self.node_type = node_type
        self.virtual_channel_probs = virtual_channel_probs

    def run_fast(self, channel_probs):
        n = int(np.log2(self.N))
        node_state = np.zeros(self.N - 1)       # node state
        depth = 0
        node = 0
        done = False
        I_original = mutual_information(0.5 * channel_probs) * self.N
        I_leaf = 0
        while done == False:
            if depth == n:
                node = node // 2
                depth -= 1
            else:
                node_posi = int(2 ** depth - 1 + node)  # node index in the binary tree

                if node_state[node_posi] == 0:

                    # R0 node
                    if self.node_type[node_posi] == 0:
                        temp = 2 ** (n - depth)
                        probs = self.virtual_channel_probs[depth, temp * node: temp * (node + 1)]
                        for i in range(temp):
                            I_leaf += mutual_information(0.5 * probs[i])
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    # R1 node
                    if self.node_type[node_posi] == 1:
                        temp = 2 ** (n - depth)
                        probs = self.virtual_channel_probs[depth, temp * node: temp * (node + 1)]
                        for i in range(temp):
                            I_leaf += mutual_information(0.5 * probs[i])
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    # REP node
                    if self.node_type[node_posi] == 2:
                        temp = 2 ** (n - depth)
                        probs = self.virtual_channel_probs[depth, temp * node: temp * (node + 1)]
                        for i in range(temp):
                            I_leaf += mutual_information(0.5 * probs[i])
                        node = node // 2
                        depth -= 1
                        continue

                    # SPC node
                    if self.node_type[node_posi] == 3:
                        temp = 2 ** (n - depth)
                        probs = self.virtual_channel_probs[depth, temp * node: temp * (node + 1)]
                        for i in range(temp):
                            I_leaf += mutual_information(0.5 * probs[i])
                        node = node // 2
                        depth -= 1
                        continue

                    temp = 2 ** (n - depth)
                    node *= 2
                    depth += 1
                    if depth == n:
                        probs = self.virtual_channel_probs[depth, temp * node: temp * (node + 1)]
                        I_leaf += mutual_information(probs[0].transpose())
                    node_state[node_posi] = 1

                elif node_state[node_posi] == 1:
                    temp = 2 ** (n - depth)
                    node = 2 * node + 1
                    depth += 1
                    if depth == n:
                        probs = self.virtual_channel_probs[depth, temp * node: temp * (node + 1)]
                        I_leaf += mutual_information(probs[0].transpose())
                    node_state[node_posi] = 2

                else:
                    if node == 0 and depth == 0:
                        done = True
                    else:
                        node = node // 2
                        depth -= 1

        return I_original, I_leaf

    def run_regular(self, channel_probs):
        n = int(np.log2(self.N))
        I_original = mutual_information(0.5 * channel_probs) * self.N
        I_leaf = 0
        node_state = np.zeros(self.N - 1)       # node state
        depth = 0
        node = 0
        done = False

        while done == False:
            if depth == n:
                if node == self.N - 1:
                    done = True
                else:
                    node = node // 2
                    depth -= 1
            else:
                node_posi = 2 ** depth - 1 + node  # node index in the binary tree

                if node_state[node_posi] == 0:  # 0 means this node is first achieved in the traversal, calc f value
                    temp = 2 ** (n - depth)
                    node *= 2
                    depth += 1
                    temp //= 2
                    if depth == n:
                        probs = self.virtual_channel_probs[-1, temp * node: temp * (node + 1)]
                        I_leaf += mutual_information(0.5 * probs[0])
                    node_state[node_posi] = 1

                elif node_state[node_posi] == 1:
                    temp = 2 ** (n - depth)
                    node = 2 * node + 1
                    depth += 1
                    temp //= 2
                    if depth == n:
                        probs = self.virtual_channel_probs[-1, temp * node: temp * (node + 1)]
                        I_leaf += mutual_information(0.5 * probs[0])
                    node_state[node_posi] = 2

                else:
                    node = node // 2
                    depth -= 1

        return I_original, I_leaf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--quantize_decoder", type=int, default=8)
    parser.add_argument("--quantize_channel_uniform", type=int, default=128)
    parser.add_argument("--quantize_channel_MMI", type=int, default=8)
    parser.add_argument("--is_quantize_channel_MMI", type=bool, default=True)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    rate = K / N
    print("N = {:d}, K = {:d}, R = {:.2f}".format(N, K, rate))

    # quantization parameter
    quantize_decoder = args.quantize_decoder
    is_quantize_channel_MMI = args.is_quantize_channel_MMI
    quantize_channel_MMI = args.quantize_channel_MMI
    quantize_channel_uniform = args.quantize_channel_uniform
    if is_quantize_channel_MMI:
        load_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, quantize_channel_MMI, quantize_decoder)
        print("channel uniform quantization level = {:d}\nchannel MMI quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel_uniform, quantize_channel_MMI, quantize_decoder))
    else:
        load_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelUniform".format(N, K, quantize_channel_uniform,
                                                                                        quantize_decoder)
        print("channel uniform quantization level = {:d}\ndecoder quantization level = {:d}".format(
            quantize_channel_uniform, quantize_decoder))

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozen_indicator, message_indicator = constructor.PW()  # PW code construction

    # node type identification
    node_identifier = NodeIdentifier(N, K, frozenbits, msgbits, use_new_node=False)
    node_type = node_identifier.run().astype(np.int32)

    # simulation parameter configuration
    EbN0dBTest = [0, 1, 2, 3, 4, 5]

    for EbN0dB in EbN0dBTest:

        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0

        if is_quantize_channel_MMI:
            highest = 1 + 3 * sigma
            lowest = -1 - 3 * sigma
            ChannelQuantizer = MMIQuantizer(px1=0.5, px_minus1=0.5)
            pyx1, interval_x = channel_transition_probability_table(quantize_channel_uniform, lowest, highest, 1, sigma)
            pyx_minus1, _ = channel_transition_probability_table(quantize_channel_uniform, lowest, highest, -1, sigma)
            joint_prob = np.zeros((2, quantize_channel_uniform)).astype(np.float32)
            joint_prob[0] = pyx1
            joint_prob[1] = pyx_minus1
            channel_lut = ChannelQuantizer.find_opt_quantizer_AWGN(joint_prob, quantize_channel_MMI)
            pzx = np.zeros((2, int(quantize_channel_MMI)))
            for i in range(int(quantize_channel_MMI)):
                begin = channel_lut[i]
                end = channel_lut[i + 1]
                pzx[0, i] = np.sum(pyx1[begin:end])
                pzx[1, i] = np.sum(pyx_minus1[begin:end])
        else:
            pzx = np.ones((2, quantize_channel_uniform))
            pzx[0], _ = channel_transition_probability_table(quantize_channel_uniform, -2, 2, -1, sigma)
            pzx[1], interval_x = channel_transition_probability_table(quantize_channel_uniform, -2, 2, 1, sigma)
            channel_lut = None

        load_path_virtual_channel_probs = os.path.join(load_dir, "Probs_EbN0dB={:d}.pkl".format(EbN0dB))
        with open(load_path_virtual_channel_probs, "rb") as f:
            virtual_channel_probs = pkl.load(f)

        capacity_calculator = EqivalentCapacity(N, K, frozenbits, msgbits, node_type, virtual_channel_probs)

        I_original_regular, I_leaf_regular = capacity_calculator.run_regular(pzx)
        I_original_fast, I_leaf_fast = capacity_calculator.run_fast(pzx)

        print("Eb/N0 = {:d}".format(EbN0dB))
        print("capacity for regular discrete decoder:")
        print("I_channel = {:f}, I_eqivalent = {:f}".format(I_original_regular, I_leaf_regular))
        print("capacity for fast discrete decoder:")
        print("I_channel = {:f}, I_eqivalent = {:f}".format(I_original_fast, I_leaf_fast))
