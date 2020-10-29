import numpy as np
from PolarDecoder.Decoder.SCLDecoder import SCLDecoder
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import argparse
import os
from tqdm import tqdm

def llr(inp, sigma):
    return (2 / sigma ** 2) * inp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--L", type=int, default=8)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N      # code length
    K = args.K      # information bits length
    L = args.L      # list size
    R = K / N       # code rate

    # code construction
    constructor = PolarCodeConstructor(N, K, "reliable sequence.txt")
    frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction

    print("N = {:d}, K = {:d}, L = {:d},  R = {:.2f}".format(N, K, L, R))

    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozen_bits=frozenbits, msg_bits=msgbits)
    polar_decoder = SCLDecoder(N, K, L, frozenbits_indicator, messagebits_indicator)
    # simulation parameter configuration
    MaxBlock = 10**6
    EbN0dB_test = [0, 1, 2, 3, 4, 5]
    ber = []
    bler = []
    total_block = 0
    for EbN0dB in EbN0dB_test:
        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        print("EbN0 = {:d}".format(EbN0dB))
        sigma = np.sqrt(1/(2*R*EbN0))  # Gaussian noise variance for current EbN0
        Nbiterrs = 0
        Nblkerrs = 0
        Nblocks = 0

        pbar = tqdm(range(MaxBlock))

        for blk in pbar:

            msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for CRC-SCL
            cword = polar_encoder.non_system_encoding(msg)  # generate non systematic polar codeword

            bpsksymbols = 1 - 2 * cword  # BPSK modulation

            receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

            receive_symbols_llr = llr(receive_symbols, sigma)  # calc channel llr

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

        total_block += Nblocks

        # calc BER & FER after every Eb/N0 simulation end
        if Nblocks == MaxBlock:
            # calc BER and FER in the given EbN0
            BER_sim = Nbiterrs / (K * Nblocks)
            FER_sim = Nblkerrs / Nblocks
            print("EbN0(dB):{:.1f}, BER:{:f}, FER:{:f}".format(EbN0dB, BER_sim, FER_sim))
            ber.append(BER_sim)
            bler.append(FER_sim)

    # save BER/BLER
    ber_save_dir = "simulation_result/BER/SCL"
    bler_save_dir = "simulation_result/BLER/SCL"
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
