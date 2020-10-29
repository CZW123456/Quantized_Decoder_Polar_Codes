import numpy as np
from QuantizeDecoder.MMISC import continous2discret, channel_transition_probability_table
from PolarDecoder.Decoder.FastSCLUTDecoder import FastSCLUTDecoder
from quantizers.quantizer.MMI import MMIQuantizer
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
from ConventionalDecoder.IdentifyNodes import NodeIdentifier
import time
import argparse
import os
import pickle as pkl
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--quantize_decoder", type=int, default=32)
    parser.add_argument("--quantize_channel_uniform", type=int, default=128)
    parser.add_argument("--quantize_channel_MMI", type=int, default=32)
    parser.add_argument("--is_quantize_channel_MMI", type=bool, default=True)
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
    if is_quantize_channel_MMI:
        load_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, quantize_channel_MMI, quantize_decoder)
        print("channel uniform quantization level = {:d}\nchannel MMI quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel_uniform, quantize_channel_MMI, quantize_decoder))
    else:
        load_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelUniform".format(N, K, quantize_channel_uniform, quantize_decoder)
        print("channel uniform quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel_uniform, quantize_decoder))

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction
    # frozenbits, msgbits = constructor.GA(2.5)  # GA code construction

    # node type identification
    node_identifier = NodeIdentifier(N, K, frozenbits, msgbits, False)
    node_type = node_identifier.run().astype(np.int32)
    # print("Binary decode tree traverse path with prunning")
    # num_node_activate = node_identifier.show_traverse_path(node_type)
    # print("{:d} nodes are activated in the tree".format(num_node_activate))

    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozenbits, msgbits)

    # simulation parameter configuration
    MaxBlock = 10**6
    EbN0dBtest = [0, 1, 2, 3, 4, 5]
    # BER/FER
    ber = []
    fer = []
    # timing variables
    total_decode_time = 0
    total_blocks = 0


    print("Simulation Begin...")
    for EbN0dB in EbN0dBtest:

        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1/(2*rate*EbN0))  # Gaussian noise variance for current EbN0

        print("Eb/N0 = {:0f} dB, Sigma2 = {:6f}".format(EbN0dB, sigma))

        if is_quantize_channel_MMI:
            highest = 1 + 3 * sigma
            lowest = -1 - 3 * sigma
            ChannelQuantizer = MMIQuantizer(0.5, 0.5)
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
            # Channel Uniform Quantizer
            pzx = np.ones((2, quantize_channel_uniform))
            pzx[0], _ = channel_transition_probability_table(quantize_channel_uniform, -2, 2, -1, sigma)
            pzx[1], interval_x = channel_transition_probability_table(quantize_channel_uniform, -2, 2, 1, sigma)
            channel_lut = None

        # Load precomputed LUT for every building block of polar codes
        if EbN0dB <= 3:
            designEbN0 = EbN0dB
        else:
            designEbN0 = 3
        load_path_f = os.path.join(load_dir, "LUT_F_EbN0dB={:d}.pkl".format(designEbN0))
        load_path_g = os.path.join(load_dir, "LUT_G_EbN0dB={:d}.pkl".format(designEbN0))
        load_path_virtual_channel_llr = os.path.join(load_dir, "LLR_EbN0dB={:d}.pkl".format(designEbN0))
        with open(load_path_f, "rb") as f:
            lut_fs = pkl.load(f)
        with open(load_path_g, "rb") as f:
            lut_gs = pkl.load(f)
        with open(load_path_virtual_channel_llr, "rb") as f:
            virtual_channel_llrs = pkl.load(f)

        # MMI Quantized SC Decoder
        virtual_channel_llrs = virtual_channel_llrs.tolist()
        lut_fs = np.array([np.array(lut_fs[key]).astype(np.int32) for key in lut_fs.keys()])
        lut_gs = np.array([np.array(lut_gs[key]).astype(np.int32) for key in lut_gs.keys()])
        fs = []
        for ele in lut_fs:
            fs.append(ele.tolist())
        gs = []
        for ele in lut_gs:
            gs.append(ele.tolist())

        polar_decoder = FastSCLUTDecoder(N, K, frozenbits_indicator, messagebits_indicator, node_type, fs, gs, virtual_channel_llrs)

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
                    y_symbols[i] = continous2discret(y[0, i], interval_x, quantize_channel_uniform)

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

        if BER_sim < 1e-5 or FER_sim < 1e-4:
            break

    # BER / BLER save path
    ber_save_dir = "simulation_result/BER/FastSCLUT"
    bler_save_dir = "simulation_result/BLER/FastSCLUT"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)

    if is_quantize_channel_MMI:
        ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, quantize_channel_MMI, quantize_decoder))
        bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, quantize_channel_MMI, quantize_decoder))
    else:
        ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}".format(N, K, quantize_channel_MMI, quantize_decoder))
        bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}".format(N, K, quantize_channel_MMI, quantize_decoder))

    # save txt
    ber = np.array(ber)
    bler = np.array(fer)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')


if __name__ == '__main__':
    main()
