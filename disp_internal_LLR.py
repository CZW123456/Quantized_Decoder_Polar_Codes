from ConventionalDecoder import SCDecoder, CodeConstruction, encoder
import ConventionalDecoder
from PolarDecoder._cpp._libPolarDecoder import SCDecoder
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--channel_level", type=int, default=0)
    parser.add_argument("--channel_idx", type=int, default=8)
    parser.add_argument("--testing_SNR", type=float, default=2.5)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    rate = K / N

    channel_level = args.channel_level
    channel_idx = args.channel_idx
    testing_EbN0 = args.testing_SNR
    EbN0 = 10 ** (testing_EbN0 / 10)  # linear scale snr
    sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0

    # code construction and extract the mean and variance of the given bit channel
    constructor = CodeConstruction.PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozen_indicator, message_indicator = constructor.GA(sigma)  # GA code construction
    E_LLR = constructor.E_LLR[channel_level, channel_idx - 1]
    D_LLR = 2 * E_LLR
    print("N = {:d}, K = {:d}, R = {:.2f}".format(N, K, rate))

    # initialize encoder and decoder
    polar_encoder = encoder.PolarEncoder(N, K, frozenbits, msgbits)
    py_polar_decoder = ConventionalDecoder.SCDecoder.SCDecoder(N, K, frozenbits, msgbits, channel_level, channel_idx)
    polar_decoder = SCDecoder(N, K, frozen_indicator, message_indicator, channel_level, channel_idx)
    # simulation parameter configuration
    MaxBlock = 5*10**5

    llrs = []

    pbar = tqdm(range(MaxBlock))
    for blk in pbar:

        msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for valina SC

        cword = polar_encoder.non_system_encoding(msg)

        bpsksymbols = 1 - 2 * cword  # BPSK modulation

        receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

        receive_symbols_llr = (2/sigma**2)*receive_symbols

        decoded_bits, llr_wanted = polar_decoder.decode(receive_symbols_llr)  # valina SC Decoder
        llrs.append(llr_wanted)

    # display LLR histogram
    llrs = np.array(llrs)
    plt.hist(llrs, bins=500, density=True)
    lrange_max = E_LLR + 3 * np.sqrt(D_LLR)
    lrange_min = -E_LLR - 3 * np.sqrt(D_LLR)
    lrange = np.linspace(start=lrange_min, stop=lrange_max, num=1000)
    p_llr_theoratical_0 = 1/np.sqrt(2*np.pi*D_LLR) * np.exp(-(lrange-E_LLR)**2/(2*D_LLR))
    p_llr_theoratical_1 = 1/np.sqrt(2*np.pi*D_LLR) * np.exp(-(lrange+E_LLR)**2/(2*D_LLR))
    p_llr_theoratical = 0.5 * (p_llr_theoratical_0 + p_llr_theoratical_1)
    plt.plot(lrange, p_llr_theoratical)
    plt.show()







