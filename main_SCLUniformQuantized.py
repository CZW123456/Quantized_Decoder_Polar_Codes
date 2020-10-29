import numpy as np
from PolarDecoder.Decoder.SCLUniformQuantizedDecoder import SCLUniformQuantizedDecoder
from QuantizeDecoder.OptUniformQuantizerGaussian import OptUniformQuantizerGaussian
from QuantizeDecoder.LLRLSUniformSC import LLRLSUniformQuantizer
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import argparse
import os
from tqdm import tqdm


def llr(inp, sigma):
    return (2/sigma ** 2) * inp


def Q(llr, M, r):
    llr[np.abs(llr) <= M] = (np.floor(llr[np.abs(llr) <= M] / r).astype(int) + 1 / 2) * r
    llr[np.abs(llr) > M] = np.sign(llr[np.abs(llr) > M]) * (M - r / 2)
    return llr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--v", type=int, default=64)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    L = args.L  # list size
    v = args.v  # quantization level
    rate = K / N

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozen_indicator, message_indicator = constructor.PW()  # PW code construction
    print("N = {:d}, K = {:d}, L = {:d}, R = {:.2f}, v = {:d}".format(N, K, L, rate, v))

    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozenbits, msgbits)
    Q_channel = OptUniformQuantizerGaussian(v)
    Q_decoder = LLRLSUniformQuantizer(N, v)
    # simulation parameter configuration
    MaxBlock = 10**6
    EbN0dBTest = [0, 1, 2, 3, 4, 5]
    # BER/FER
    ber = []
    fer = []
    # timing variables
    total_decode_time = 0
    total_blocks = 0

    design_SNR = 0
    EbN0 = 10 ** (design_SNR / 10)  # linear scale snr
    sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0
    decoder_r_f, decoder_r_g = Q_decoder.generate_uniform_quantizers(sigma)
    decoder_r_f = decoder_r_f.tolist()
    decoder_r_g = decoder_r_g.tolist()
    polar_decoder = SCLUniformQuantizedDecoder(N, K, L, frozen_indicator, message_indicator, decoder_r_f, decoder_r_g, v)

    for EbN0dB in EbN0dBTest:
        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0

        mu_llr = 2/sigma**2
        sigma2_llr = 2*mu_llr
        opt_r_channel = Q_channel.find_optimal_interval_bimodal_Gaussian(mu_llr, sigma2_llr, 30)
        M_channel = (v//2 - 1)*opt_r_channel

        Nbiterrs = 0
        Nblkerrs = 0
        Nblocks = 0

        pbar = tqdm(range(MaxBlock))
        for _ in pbar:

            msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for valina SC

            cword = polar_encoder.non_system_encoding(msg)

            bpsksymbols = 1 - 2 * cword  # BPSK modulation

            receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

            receive_symbols_llr = llr(receive_symbols, sigma)  # calc channel llr
            quantized_llr = Q(receive_symbols_llr, M_channel, opt_r_channel)
            decoded_bits = polar_decoder.decode(quantized_llr)  # valina SC Decoder

            # calc error statistics
            Nbiterrs += np.sum(msg != decoded_bits)
            Nblkerrs += np.any(msg != decoded_bits)

            if Nblkerrs > 1000:
                BER_sim = Nbiterrs / (K * Nblocks)
                FER_sim = Nblkerrs / Nblocks
                print("EbN0(dB):{:.1f}, BER:{:f}, FER:{:f}".format(EbN0dB, BER_sim, FER_sim))
                ber.append(BER_sim)
                fer.append(FER_sim)
                break
            Nblocks += 1
            pbar.set_description("Err Bits = {:d}, Err Frame = {:d}".format(Nbiterrs, Nblkerrs))

        if Nblocks == MaxBlock:
            # calc BER and FER in the given EbN0
            BER_sim = Nbiterrs / (K * Nblocks)
            FER_sim = Nblkerrs / Nblocks
            print("EbN0(dB):{:.1f}, BER:{:f}, FER:{:f}".format(EbN0dB, BER_sim, FER_sim))
            ber.append(BER_sim)
            fer.append(FER_sim)
        total_blocks += Nblocks

    # save BER/BLER
    ber_save_dir = "simulation_result/BER/SCLUniformQuantized"
    bler_save_dir = "simulation_result/BLER/SCLUniformQuantized"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)
    ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_L{:d}_v{:d}".format(N, K, L, v))
    bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_L{:d}_v{:d}".format(N, K, L, v))
    ber = np.array(ber)
    bler = np.array(fer)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')


if __name__ == '__main__':
    main()
