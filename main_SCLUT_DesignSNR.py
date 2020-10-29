import numpy as np
import os
from QuantizeDecoder.MMISC import continous2discret, channel_transition_probability_table
from QuantizeDecoder.SCLUTDecoder import SCLUTDecoder
from quantizers.quantizer.MMI import MMIQuantizer
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import time
import argparse
import pickle as pkl
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--quantize_decoder", type=int, default=16)
    parser.add_argument("--quantize_channel_uniform", type=int, default=128)
    parser.add_argument("--quantize_channel_MMI", type=int, default=16)
    parser.add_argument("--is_quantize_channel_MMI", type=bool, default=True)
    parser.add_argument("--design_EbN0", type=int, default=0)
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
    design_EbN0 = args.design_EbN0
    if is_quantize_channel_MMI:
        load_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, quantize_channel_MMI, quantize_decoder)
        print("channel uniform quantization level = {:d}\nchannel MMI quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel_uniform, quantize_channel_MMI, quantize_decoder))
    else:
        load_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelUniform".format(N, K, quantize_channel_uniform, quantize_decoder)
        print("channel uniform quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel_uniform, quantize_decoder))

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits = constructor.PW()  # PW code construction

    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozenbits, msgbits)

    # simulation parameter configuration
    MaxBlock = 5*10**5
    StartEbN0 = 0
    MaxEbN0dB = 7
    # BER/FER
    ber = []
    fer = []
    # timing variables
    total_decode_time = 0
    total_blocks = 0
    print("EbN0 range:[{:d}, {:d}] Number Blocks Per EbN0 = {:d}".format(StartEbN0, MaxEbN0dB, MaxBlock))

    print("Simulation Begin...")
    for EbN0dB in range(StartEbN0, MaxEbN0dB):

        EbN0 = 10 ** (EbN0dB/10)  # linear scale snr
        sigma = np.sqrt(1/(2*rate*EbN0))  # Gaussian noise variance for current EbN0

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

        # Load precomputed LUT for every building block of polar codes
        load_path_f = os.path.join(load_dir, "LUT_F_EbN0dB={:d}.pkl".format(design_EbN0))
        load_path_g = os.path.join(load_dir, "LUT_G_EbN0dB={:d}.pkl".format(design_EbN0))
        load_path_virtual_channel_llr = os.path.join(load_dir, "LLR_EbN0dB={:d}.pkl".format(design_EbN0))
        with open(load_path_f, "rb+") as f:
            lut_fs = pkl.load(f)
        with open(load_path_g, "rb+") as f:
            lut_gs = pkl.load(f)
        with open(load_path_virtual_channel_llr, "rb") as f:
            virtual_channel_llrs = pkl.load(f)

        # MMI Quantized SC Decoder
        polar_decoder = SCLUTDecoder(N, K, frozenbits, msgbits, lut_fs, lut_gs, virtual_channel_llrs)

        Nbiterrs = 0
        Nblkerrs = 0
        Nblocks = 0

        pbar = tqdm(range(MaxBlock))

        for _ in pbar:

            msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for valina SC

            cword = polar_encoder.non_system_encoding(msg)

            bpsksymbols = 1 - 2 * cword  # BPSK modulation

            y = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

            y_symbols = np.zeros(N)
            if is_quantize_channel_MMI:
                for i in range(N):
                    y_symbols[i] = continous2discret(y[0, i], interval_x[channel_lut], quantize_channel_MMI - 1)
            else:
                for i in range(N):
                    y_symbols[i] = continous2discret(y[0, i], interval_x, quantize_channel_uniform - 1)

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
    ber_save_dir = "simulation_result/BER/SCLUT_SingleLUT"
    bler_save_dir = "simulation_result/BLER/SCLUT_SingleLUT"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)

    if is_quantize_channel_MMI:
        ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_DesignEbN0{:d}dB_ChannelMMI".format(N, K, quantize_channel_MMI, quantize_decoder, design_EbN0))
        bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_DesignEbN0{:d}dB_ChannelMMI".format(N, K, quantize_channel_MMI, quantize_decoder, design_EbN0))
    else:
        ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_DesignEbN0{:d}dB".format(N, K, quantize_channel_MMI, quantize_decoder, design_EbN0))
        bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_DesignEbN0{:d}dB".format(N, K, quantize_channel_MMI, quantize_decoder, design_EbN0))

    # save txt
    ber = np.array(ber)
    bler = np.array(fer)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')


if __name__ == '__main__':
    main()
