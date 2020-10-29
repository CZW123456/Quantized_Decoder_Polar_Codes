import numpy as np
from QuantizeDecoder.MMISC import continous2discret, channel_transition_probability_table
from PolarDecoder.Decoder.CASCLLUTDecoder import CASCLLUTDecoder
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CRC import CRC
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import argparse
import os
import pickle as pkl
from tqdm import tqdm
from quantizers.quantizer.MMI import MMIQuantizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--quantize_decoder", type=int, default=32)
    parser.add_argument("--quantize_channel_uniform", type=int, default=128)
    parser.add_argument("--quantize_channel", type=int, default=32)
    args = parser.parse_args()
    crc_n = 24  # CRC check code length
    crc_p = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]  # CRC generator polynomial
    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    A = args.K - crc_n
    L = args.L  # list size
    rate = K / N
    rate_noise = A/N
    print("N = {:d}, K = {:d}, L = {:d}, R = {:.2f}".format(N, K, L, rate))

    # quantization parameter
    quantize_decoder = args.quantize_decoder
    quantize_channel = args.quantize_channel
    quantize_channel_uniform = args.quantize_channel_uniform

    load_dir = "./LUTs_LLROptLS/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}".format(N, K, quantize_channel, quantize_decoder)
    print("channel uniform quantization level = {:d}\nchannel MMI quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel_uniform, quantize_channel, quantize_decoder))


    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction

    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozenbits, msgbits)
    crc = CRC(crc_n=crc_n, crc_p=crc_p)
    # simulation parameter configuration
    MaxBlock = 10**6
    EbN0dBTest = [0, 1, 2, 3, 4, 5]
    # BER/FER
    ber = []
    fer = []
    # timing variables
    total_decode_time = 0
    total_blocks = 0
    print("EbN0 range:[{:d}, {:d}] Number Blocks Per EbN0 = {:d}".format(EbN0dBTest[0], EbN0dBTest[-1], MaxBlock))
    print("Simulation Begin...")
    for EbN0dB in EbN0dBTest:

        designEbN0dB = EbN0dB

        EbN0 = 10**(EbN0dB/10)  # linear scale snr
        sigma = np.sqrt(1/(2*rate_noise*EbN0))  # Gaussian noise variance for current EbN0

        print("Eb/N0 = {:0f} dB, Sigma2 = {:6f}".format(EbN0dB, sigma**2))

        E_LLR = 2 / (sigma ** 2)
        V_LLR = 2 * E_LLR
        D_LLR = np.sqrt(V_LLR)
        highest = E_LLR + 3 * D_LLR
        lowest = -E_LLR - 3 * D_LLR
        ChannelQuantizer = MMIQuantizer(px1=0.5, px_minus1=0.5)
        pyx1, interval_x, = channel_transition_probability_table(quantize_channel_uniform, lowest, highest, E_LLR,
                                                                 D_LLR)
        pyx_minus1, _, = channel_transition_probability_table(quantize_channel_uniform, lowest, highest, -E_LLR, D_LLR)
        quanta_channel_uniform = interval_x[:-1] + 0.5 * (interval_x[1] - interval_x[0])
        joint_prob = np.zeros((2, quantize_channel_uniform)).astype(np.float32)
        joint_prob[0] = pyx1
        joint_prob[1] = pyx_minus1
        channel_lut = ChannelQuantizer.find_opt_quantizer_AWGN(joint_prob, quantize_channel)
        quanta_channel = np.zeros(quantize_channel)  # the quanta of each quantization interval after MMI compression of the channel LLR density
        channel_llr_density = np.zeros(quantize_channel)
        for i in range(int(quantize_channel)):
            begin = channel_lut[i]
            end = channel_lut[i + 1]
            pxz = 0.5 * (pyx1[begin:end] + pyx_minus1[begin:end])
            quanta_channel[i] = np.sum(quanta_channel_uniform[begin:end] * pxz) / np.sum(pxz)  # the quanta should be the mass center of the quantization interval
            channel_llr_density[i] = np.sum(pxz)

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

        polar_decoder = CASCLLUTDecoder(N, K, A, L, frozenbits_indicator, messagebits_indicator, fs, gs, llr_quanta)


        Nbiterrs = 0
        Nblkerrs = 0
        Nblocks = 0

        pbar = tqdm(range(MaxBlock))

        for _ in pbar:

            msg = np.random.randint(low=0, high=2, size=A)  # generate 0-1 msg bits for valina SC
            msg_crc, _ = crc.encode(msg)  # CRC encoded msg

            cword = polar_encoder.non_system_encoding(msg_crc)

            bpsksymbols = 1 - 2 * cword  # BPSK modulation

            y = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

            y *= (2/sigma**2)

            llr_symbols = np.zeros(N).astype(np.int32)

            for i in range(N):
                llr_symbols[i] = continous2discret(y[0, i], interval_x[channel_lut], quantize_channel - 1)


            decoded_bits = polar_decoder.decode(llr_symbols)  # valina SC Decoder

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

    # BER / BLER save path
    ber_save_dir = "simulation_result/BER/CA-SCLLLRLUT"
    bler_save_dir = "simulation_result/BLER/CA-SCLLLRLUT"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)
    ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_L{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, L, quantize_channel, quantize_decoder))
    bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_L{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, L, quantize_channel, quantize_decoder))

    # save txt
    ber = np.array(ber)
    bler = np.array(fer)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')

if __name__ == '__main__':
    main()