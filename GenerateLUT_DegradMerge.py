import numpy as np
from BitChannelMerge.utils import get_AWGN_transition_probability_degrading
import pickle as pkl
import os
from BitChannelMerge.DegradingMerge import DegradeMerger
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--mu", type=int, default=16)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    rate = K / N
    mu = args.mu
    print("N = {:d}, K = {:d}, mu = {:d}".format(N, K, mu))

    save_dir = "./LUTs_DegradMerge/N{:d}_K{:d}_Mu{:d}".format(N, K, mu)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    EbN0dBTest = [0, 1, 2, 3, 4, 5]
    print("EbN0 range:[{:d}, {:d}].".format(EbN0dBTest[0], EbN0dBTest[-1]))
    for EbN0dB in EbN0dBTest:
        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0
        print("Eb/N0 = {:0f} dB, Sigma2 = {:6f}".format(EbN0dB, sigma))
        W_AWGN, y_interval = get_AWGN_transition_probability_degrading(sigma, mu // 2)
        Merger = DegradeMerger(mu, N, W_AWGN)
        virtual_channel_llrs, Pe, LUT_Fs, LUT_Gs = Merger.bit_channel_degrading_run()

        save_path_f = os.path.join(save_dir, "LUT_F_EbN0dB={:d}.pkl".format(EbN0dB))
        save_path_g = os.path.join(save_dir, "LUT_G_EbN0dB={:d}.pkl".format(EbN0dB))
        save_path_virtual_channel_llr = os.path.join(save_dir, "LLR_EbN0dB={:d}.pkl".format(EbN0dB))
        with open(save_path_f, "wb") as f:
            pkl.dump(LUT_Fs, f)
        with open(save_path_g, "wb") as g:
            pkl.dump(LUT_Gs, g)
        with open(save_path_virtual_channel_llr, "wb") as v:
            pkl.dump(virtual_channel_llrs, v)

        print("Save LUTs for f function to : {:s}".format(save_path_f))
        print("Save LUTs for g function to : {:s}".format(save_path_g))
        print("Save virtual channel llr to : {:s}".format(save_path_virtual_channel_llr))



