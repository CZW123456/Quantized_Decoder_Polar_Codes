import numpy as np
from QuantizeDecoder.MMISC import channel_transition_probability_table
from quantizers.quantizer.MMI import MMIQuantizer
from QuantizeDecoder.LLRQuantizedSC import LLRQuantizerSC
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import pickle as pkl
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--v", type=int, default=16)
    parser.add_argument("--quantize_channel_uniform", type=int, default=128)

    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    rate = K / N
    print("N = {:d}, K = {:d}, R = {:.2f}".format(N, K, rate))

    # quantization parameter
    quantize_decoder = args.v
    quantize_channel = args.v
    quantize_channel_uniform = args.quantize_channel_uniform

    save_dir = "./LUTs_LLROptLS/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}".format(N, K, quantize_channel, quantize_decoder)
    print("Channel quantization level = {:d}\nDecoder quantization level = {:d}".format(quantize_channel, quantize_decoder))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, _, _ = constructor.PW()  # PW code construction

    # simulation parameter configuration
    EbN0dBTest = [0, 1, 2, 3, 4, 5]

    print("EbN0 range:[{:d}, {:d}].".format(EbN0dBTest[0], EbN0dBTest[-1]))
    for EbN0dB in EbN0dBTest:

        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0

        print("Eb/N0 = {:0f} dB, Sigma2 = {:6f}".format(EbN0dB, sigma))


        # Channel Quantizer
        E_LLR = 2 / (sigma ** 2)
        V_LLR = 2 * E_LLR
        D_LLR = np.sqrt(V_LLR)
        highest = E_LLR + 3 * D_LLR
        lowest = -E_LLR - 3 * D_LLR
        ChannelQuantizer = MMIQuantizer(px1=0.5, px_minus1=0.5)
        pyx1, interval_x, = channel_transition_probability_table(quantize_channel_uniform, lowest, highest, E_LLR, D_LLR)
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

        # Generate LUT for every building block of polar codes
        print("Generate SC MMI Quantizer...")
        SCQuantizer = LLRQuantizerSC(N, K, quantize_decoder)
        llr_density, llr_quanta, lut_fs, lut_gs = SCQuantizer.generate_LUTs(channel_llr_density=channel_llr_density,
                                                                            channel_llr_quanta=quanta_channel)
        print("LUT Generation Finished...")

        save_path_f = os.path.join(save_dir, "LUT_F_EbN0dB={:d}.pkl".format(EbN0dB))
        save_path_g = os.path.join(save_dir, "LUT_G_EbN0dB={:d}.pkl".format(EbN0dB))
        save_path_llr_quanta = os.path.join(save_dir, "LLRQuanta_EbN0dB={:d}.pkl".format(EbN0dB))
        save_path_llr_density = os.path.join(save_dir, "LLRDensity_EbN0dB={:d}.pkl".format(EbN0dB))
        with open(save_path_f, "wb") as f:
            pkl.dump(lut_fs, f)
        with open(save_path_g, "wb") as f:
            pkl.dump(lut_gs, f)
        with open(save_path_llr_quanta, "wb") as f:
            pkl.dump(llr_quanta, f)
        with open(save_path_llr_density, "wb") as f:
            pkl.dump(llr_density, f)

        print("Save LUTs for f function to : {:s}".format(save_path_f))
        print("Save LUTs for g function to : {:s}".format(save_path_g))
        print("Save LLR Quantas to : {:s}".format(save_path_llr_quanta))
        print("Save LLR Density to : {:s}".format(save_path_llr_density))

if __name__ == '__main__':
    main()