import numpy as np
from QuantizeDecoder.QDensityEvolution_MMI import continous2discret
from QuantizeDecoder.LLRMIQuantizer import LLRMIQuantizer
from PolarDecoder.Decoder.SCLUTDecoder import SCLUTDecoder
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import argparse
import os
import pickle as pkl
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--quantize_decoder", type=int, default=32)
    parser.add_argument("--quantize_channel_uniform", type=int, default=128)
    parser.add_argument("--quantize_channel", type=int, default=32)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    rate = K / N
    print("N = {:d}, K = {:d}, R = {:.2f}".format(N, K, rate))

    # quantization parameter
    quantize_decoder = args.quantize_decoder
    quantize_channel = args.quantize_channel

    load_dir = "./LUTs_LLROptLS/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}".format(N, K, quantize_channel, quantize_decoder)
    print("channel uniform quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel, quantize_decoder))

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozen_indicator, message_indicator = constructor.PW()  # PW code construction

    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozenbits, msgbits)

    # simulation parameter configuration
    MaxBlock = 5*10**5
    EbN0dBTest = [0, 1, 2, 3, 4, 5]
    # BER/FER
    ber = []
    fer = []
    # timing variables
    total_blocks = 0

    print("Simulation Begin...")
    for EbN0dB in EbN0dBTest:

        designEbN0dB = 3

        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma_awgn = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0

        # Channel Quantizer
        mu = 2 / sigma_awgn ** 2
        sigma2_llr = 2 * mu
        begin = -mu - 3 * np.sqrt(sigma2_llr)
        end = mu + 3 * np.sqrt(sigma2_llr)
        quantizer = LLRMIQuantizer(quantize_channel)
        boundary, reconstruction, plx = quantizer.find_quantizer_awgn(mu, sigma2_llr, begin, end)


        load_path_f = os.path.join(load_dir, "LUT_F_EbN0dB={:d}.pkl".format(designEbN0dB))
        load_path_g = os.path.join(load_dir, "LUT_G_EbN0dB={:d}.pkl".format(designEbN0dB))
        load_path_llr_quanta = os.path.join(load_dir, "LLRQuanta_EbN0dB={:d}.pkl".format(designEbN0dB))
        with open(load_path_f, "rb+") as f:
            lut_fs = pkl.load(f)
        with open(load_path_g, "rb+") as f:
            lut_gs = pkl.load(f)
        with open(load_path_llr_quanta, "rb") as f:
            llr_quanta = pkl.load(f)

        llr_quanta = llr_quanta.tolist()
        lut_fs = np.array([np.array(lut_fs[key]).astype(np.int32) for key in lut_fs.keys()])
        lut_gs = np.array([np.array(lut_gs[key]).astype(np.int32) for key in lut_gs.keys()])
        fs = []
        for ele in lut_fs:
            fs.append(ele.tolist())
        gs = []
        for ele in lut_gs:
            gs.append(ele.tolist())

        # Quantized SC Decoder
        polar_decoder = SCLUTDecoder(N, K, frozen_indicator, message_indicator, lut_fs, lut_gs, llr_quanta)

        Nbiterrs = 0
        Nblkerrs = 0
        Nblocks = 0

        pbar = tqdm(range(MaxBlock))

        for _ in pbar:

            msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for valina SC

            cword = polar_encoder.non_system_encoding(msg)

            bpsksymbols = 1 - 2 * cword  # BPSK modulation

            y = bpsksymbols + np.random.normal(loc=0, scale=sigma_awgn, size=(1, N))  # AWGN noisy channel

            # obtain LLR under AWGN
            llr = y * 2 / (sigma_awgn**2)

            llr_symbols = np.zeros(N).astype(np.int32)

            for i in range(N):
                llr_symbols[i] = continous2discret(llr[0, i], boundary, quantize_channel - 1)

            decoded_bits = polar_decoder.decode(llr_symbols)  # valina SC Decoder

            # calc error statistics
            Nbiterrs += np.sum(msg != decoded_bits)
            Nblkerrs += np.any(msg != decoded_bits)

            if Nblkerrs > 1000:
                BER_sim = Nbiterrs / (K * Nblocks)
                FER_sim = Nblkerrs / Nblocks
                print("EbN0(dB):{:.1f}, Sigma2 = {:6f},  BER:{:f}, FER:{:f}".format(EbN0dB, sigma_awgn, BER_sim, FER_sim))
                ber.append(BER_sim)
                fer.append(FER_sim)
                break
            Nblocks += 1
            pbar.set_description("Err Bits = {:d}, Err Frame = {:d}".format(Nbiterrs, Nblkerrs))

        if Nblocks == MaxBlock:
            # calc BER and FER in the given EbN0
            BER_sim = Nbiterrs / (K * Nblocks)
            FER_sim = Nblkerrs / Nblocks
            print("EbN0(dB):{:.1f}, Sigma2 = {:6f}, BER:{:f}, FER:{:f}".format(EbN0dB, sigma_awgn, BER_sim, FER_sim))
            ber.append(BER_sim)
            fer.append(FER_sim)
        total_blocks += Nblocks

    # BER / BLER save path
    ber_save_dir = "simulation_result/BER/SCLLRMILUT"
    bler_save_dir = "simulation_result/BLER/SCLLRMILUT"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)


    ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K,
                                                                                                         quantize_channel,
                                                                                                         quantize_decoder))

    bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K,
                                                                                                           quantize_channel,
                                                                                                           quantize_decoder))
    # save txt
    ber = np.array(ber)
    bler = np.array(fer)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')


if __name__ == '__main__':
    main()