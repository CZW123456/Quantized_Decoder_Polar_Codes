import numpy as np
import ConventionalDecoder
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CRC import CRC
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import argparse
import os
from tqdm import tqdm
from PolarDecoder.Decoder.CASCLDecoder import CASCLDecoder

def llr(inp, sigma):
    return (2 / sigma ** 2) * inp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--L", type=int, default=8)
    args = parser.parse_args()
    crc_n = 24  # CRC check code length
    crc_p = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]  # CRC generator polynomial
    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    A = args.K - crc_n  # true information length
    L = 8  # list size
    rate = K / N
    rate_noise = A / N

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction
    crc = CRC(crc_n=crc_n, crc_p=crc_p)
    print("N = {:d}, K = {:d}, R = {:.2f}".format(N, K, rate))

    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozen_bits=frozenbits, msg_bits=msgbits)
    polar_decoder_py = ConventionalDecoder.CASCLDecoder.CASCLDecoder(N, K, A, L, frozenbits, msgbits, crc_n, crc_p)
    polar_decoder = CASCLDecoder(N, K, A, L, frozenbits_indicator, messagebits_indicator)
    # simulation parameter configuration
    MaxBlock = 10**6
    EbN0dB_test = [0, 1, 2, 3, 4, 5]
    ber = []
    bler = []
    for EbN0dB in EbN0dB_test:
        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate_noise * EbN0))  # Gaussian noise variance for current EbN0
        Nbiterrs = 0
        Nblkerrs = 0
        Nblocks = 0
        total_time = 0
        pbar = tqdm(range(MaxBlock))
        for _ in pbar:

            msg = np.random.randint(low=0, high=2, size=A)  # generate 0-1 msg bits for CRC-SCL
            msg_crc, _ = crc.encode(msg)  # CRC encoded msg
            cword = polar_encoder.non_system_encoding(msg_crc)  # generate non systematic polar codeword

            bpsksymbols = 1 - 2 * cword  # BPSK modulation

            receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

            receive_symbols_llr = llr(receive_symbols, sigma)  # calc channel llr

            # decoded_bits = polar_decoder_py.decode(receive_symbols_llr)
            decoded_bits = polar_decoder.decode(receive_symbols_llr)  # CRC-SCL Decoder with Lazy Copy


            # calc error statistics
            Nbiterrs += np.sum(msg != decoded_bits)
            Nblkerrs += np.any(msg != decoded_bits)
            Nblocks += 1

            if Nblkerrs > 1000:
                BER_sim = Nbiterrs / (K * Nblocks)
                FER_sim = Nblkerrs / Nblocks
                print("EbN0(dB):{:.1f}, BER:{:f}, FER:{:f}".format(EbN0dB, BER_sim, FER_sim))
                ber.append(BER_sim)
                bler.append(FER_sim)
                break
            pbar.set_description("Err Bits = {:d}, Err Frame = {:d}".format(Nbiterrs, Nblkerrs))

        print("total time = {:.6f}".format(total_time))
        # calc BER & FER after every Eb/N0 simulation end
        if Nblocks == MaxBlock:
            # calc BER and FER in the given EbN0
            BER_sim = Nbiterrs / (K * Nblocks)
            FER_sim = Nblkerrs / Nblocks
            print("EbN0(dB):{:.1f}, BER:{:f}, FER:{:f}".format(EbN0dB, BER_sim, FER_sim))
            ber.append(BER_sim)
            bler.append(FER_sim)

    # save BER/BLER
    ber_save_dir = "simulation_result/BER/CA-SCL"
    bler_save_dir = "simulation_result/BLER/CA-SCL"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)
    ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_L{:d}".format(N, K, L))
    bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_L{:d}".format(N, K, L))
    ber = np.array(ber)
    bler = np.array(bler)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')


if __name__ == '__main__':
    main()
