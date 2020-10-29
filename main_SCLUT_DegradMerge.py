import numpy as np
from QuantizeDecoder.MMISC import continous2discret
from PolarDecoder.Decoder.SCLUTDecoder import SCLUTDecoder
import QuantizeDecoder.SCLUTDecoder as PyDecoder
from BitChannelMerge.utils import get_AWGN_transition_probability_degrading
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import time
import argparse
import os
import pickle as pkl
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--mu", type=int, default=32)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    rate = K / N
    print("N = {:d}, K = {:d}, R = {:.2f}".format(N, K, rate))

    # quantization parameter
    mu = args.mu

    load_dir = "./LUTs_DegradMerge/N{:d}_K{:d}_Mu{:d}".format(N, K, mu)

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozen_indicator, message_indicator = constructor.PW()  # PW code construction

    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozenbits, msgbits)

    # simulation parameter configuration
    MaxBlock = 10**6
    EbN0dBTest = [0, 1, 2, 3, 4, 5]
    # BER/FER
    ber = []
    fer = []
    # timing variables
    total_decode_time = 0
    total_blocks = 0

    print("Simulation Begin...")
    for EbN0dB in EbN0dBTest:

        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0
        W_AWGN, y_interval = get_AWGN_transition_probability_degrading(sigma, mu // 2)
        # Load precomputed LUT for every building block of polar codes
        if EbN0dB > 3:
            designEbN0dB = 3
        else:
            designEbN0dB = EbN0dB
        load_path_f = os.path.join(load_dir, "LUT_F_EbN0dB={:d}.pkl".format(designEbN0dB))
        load_path_g = os.path.join(load_dir, "LUT_G_EbN0dB={:d}.pkl".format(designEbN0dB))
        load_path_virtual_channel_llr = os.path.join(load_dir, "LLR_EbN0dB={:d}.pkl".format(designEbN0dB))
        with open(load_path_f, "rb+") as f:
            lut_fs = pkl.load(f)
        with open(load_path_g, "rb+") as f:
            lut_gs = pkl.load(f)
        with open(load_path_virtual_channel_llr, "rb") as f:
            virtual_channel_llrs = pkl.load(f)

        # MMI Quantized SC Decoder
        llr_tmp = virtual_channel_llrs.copy()
        virtual_channel_llrs = virtual_channel_llrs.tolist()

        lut_fs = np.array([np.array(lut_fs[key]).astype(np.int32) for key in lut_fs.keys()])
        lut_gs = np.array([np.array(lut_gs[key]).astype(np.int32) for key in lut_gs.keys()])
        fs = []
        for ele in lut_fs:
            fs.append(ele.tolist())
        gs = []
        for ele in lut_gs:
            gs.append(ele.tolist())

        polar_decoder = SCLUTDecoder(N, K, frozen_indicator, message_indicator, fs, gs, virtual_channel_llrs)
        polar_decoder_py = PyDecoder.SCLUTDecoder(N, K, frozenbits, msgbits, lut_fs, lut_gs, llr_tmp)
        Nbiterrs = 0
        Nblkerrs = 0
        Nblocks = 0

        pbar = tqdm(range(MaxBlock))

        for _ in pbar:

            msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for valina SC

            cword = polar_encoder.non_system_encoding(msg)

            bpsksymbols = 1 - 2 * cword  # BPSK modulation

            y = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

            y_symbols = np.zeros(N).astype(np.int32)

            for i in range(N):
                y_symbols[i] = continous2discret(y[0, i], y_interval, mu - 1)

            t1 = time.time()
            decoded_bits = polar_decoder.decode(y_symbols)  # valina SC Decoder
            t2 = time.time()

            total_decode_time += (t2 - t1)
            # calc error statistics
            Nbiterrs += np.sum(msg != decoded_bits)
            Nblkerrs += np.any(msg != decoded_bits)

            if Nblkerrs > 1000:
                BER_sim = Nbiterrs / (K * Nblocks)
                FER_sim = Nblkerrs / Nblocks
                print("EbN0(dB):{:.1f}, Sigma2 = {:6f},  BER:{:f}, FER:{:f}".format(EbN0dB, sigma, BER_sim, FER_sim))
                ber.append(BER_sim)
                fer.append(FER_sim)
                break
            Nblocks += 1
            pbar.set_description("Err Bits = {:d}, Err Frame = {:d}".format(Nbiterrs, Nblkerrs))

        if Nblocks == MaxBlock:
            # calc BER and FER in the given EbN0
            BER_sim = Nbiterrs / (K * Nblocks)
            FER_sim = Nblkerrs / Nblocks
            print("EbN0(dB):{:.1f}, Sigma2 = {:6f}, BER:{:f}, FER:{:f}".format(EbN0dB, sigma, BER_sim, FER_sim))
            ber.append(BER_sim)
            fer.append(FER_sim)
        total_blocks += Nblocks

    # BER / BLER save path
    ber_save_dir = "simulation_result/BER/SCLUT_DegradMerge"
    bler_save_dir = "simulation_result/BLER/SCLUT_DegradMerge"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)


    ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_Mu{:d}".format(N, K, mu))
    bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_mu{:d}".format(N, K, mu))

    # save txt
    ber = np.array(ber)
    bler = np.array(fer)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')


if __name__ == '__main__':
    main()
