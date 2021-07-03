import numpy as np
from PyIBQuantizer.inf_theory_tools import mutual_information
import argparse
from QuantizeDensityEvolution.QDensityEvolution_MMI import channel_transition_probability_table
from quantizers.quantizer.MMI import MMIQuantizer
import os
import pickle as pkl
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--quantize_decoder", type=int, default=16)
    parser.add_argument("--quantize_channel_uniform", type=int, default=128)
    parser.add_argument("--quantize_channel_MMI", type=int, default=16)
    parser.add_argument("--is_quantize_channel_MMI", type=bool, default=True)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    rate = K / N
    n = int(np.log2(N))
    print("N = {:d}, K = {:d}, R = {:.2f}".format(N, K, rate))

    # simulation parameter configuration
    designEbN0 = 3
    EbN0 = 10 ** (designEbN0 / 10)  # linear scale snr
    sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0
    nbits = [4, 5]

    capacity_loss = np.zeros((3, n + 1))

    for bit in nbits:
        load_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, 2 ** bit, 2 ** bit)
        load_path_virtual_channel_probs = os.path.join(load_dir, "Probs_EbN0dB={:d}.pkl".format(designEbN0))
        with open(load_path_virtual_channel_probs, "rb") as f:
            virtual_channel_probs = pkl.load(f)

        # capacity of AWGN channels
        highest = 1 + 3 * sigma
        lowest = -1 - 3 * sigma
        ChannelQuantizer = MMIQuantizer(px1=0.5, px_minus1=0.5)
        pyx1, interval_x = channel_transition_probability_table(128, lowest, highest, 1, sigma)
        pyx_minus1, _ = channel_transition_probability_table(128, lowest, highest, -1, sigma)
        joint_prob = np.zeros((2, 128)).astype(np.float32)
        joint_prob[0] = pyx1
        joint_prob[1] = pyx_minus1
        channel_lut = ChannelQuantizer.find_opt_quantizer_AWGN(joint_prob, 2**bit)
        pzx = np.zeros((2, int(2**bit)))
        for i in range(int(2**bit)):
            begin = channel_lut[i]
            end = channel_lut[i + 1]
            pzx[0, i] = np.sum(pyx1[begin:end])
            pzx[1, i] = np.sum(pyx_minus1[begin:end])


        capacity_loss[bit-3, 0] = mutual_information(0.5 * pzx.transpose())

        for l in range(int(np.log2(N))):
            capacity = np.zeros(N)
            for c in range(N):
                capacity[c] = mutual_information(0.5 * virtual_channel_probs[l, c].transpose())
            capacity_loss[bit-3, l + 1] = np.mean(capacity)

    np.savetxt("capacity_loss-({:d}, {:d})".format(N, K), capacity_loss, fmt="%.6f")
