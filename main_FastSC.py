import numpy as np
from PolarDecoder.Decoder.FastSCDecoder import FastSCDecoder
import ConventionalDecoder
from ConventionalDecoder.IdentifyNodes import NodeIdentifier
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import time
import argparse
from tqdm import tqdm
import os

def llr(inp, sigma):
    return (2 / sigma ** 2) * inp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=32)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    rate = K / N
    # initialize encoder and decoder

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction

    # node type identification
    node_identifier = NodeIdentifier(N, K, frozenbits, msgbits, use_new_node=False)
    node_type = node_identifier.run().astype(np.int32)

    # Polar encoder and decoder initialization
    polar_encoder = PolarEncoder(N, K, frozenbits, msgbits)
    polar_decoder_py = ConventionalDecoder.FastSCDecoder.FastSCDecoder(N, K, node_type, frozenbits, msgbits)
    polar_decoder = FastSCDecoder(N, K, frozenbits_indicator, messagebits_indicator, node_type)
    # simulation parameter configuration
    MaxBlock = 5*10**5
    EbN0dBtest = [0, 1, 2, 3, 4, 5]
    # BER/FER
    ber = []
    fer = []
    # timing variables
    total_decode_time = 0
    total_blocks = 0
    for EbN0dB in EbN0dBtest:
        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0
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

            t1 = time.time()
            decoded_bits = polar_decoder.decode(receive_symbols_llr)  # valina SC Decoder
            t2 = time.time()
            total_decode_time += (t2 - t1)
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
    ber_save_dir = "simulation_result/BER/FastSC"
    bler_save_dir = "simulation_result/BLER/FastSC"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)
    ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}".format(N, K))
    bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}".format(N, K))
    ber = np.array(ber)
    bler = np.array(fer)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')

if __name__ == '__main__':
    main()
