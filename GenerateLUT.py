import numpy as np
from QuantizeDecoder.MMISC import SCQuantizer, channel_transition_probability_table
from quantizers.quantizer.MMI import MMIQuantizer
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import pickle as pkl
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--quantize_decoder", type=int, default=16)
    parser.add_argument("--quantize_channel_uniform", type=int, default=128)
    parser.add_argument("--quantize_channel_MMI", type=int, default=16)
    parser.add_argument("--is_quantize_channel_MMI", type=bool, default=True)
    parser.add_argument("--is_save_lut", type=bool, default=True)
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
        save_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelMMI".format(N, K, quantize_channel_MMI, quantize_decoder)
        print("channel uniform quantization level = {:d}\nchannel MMI quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel_uniform, quantize_channel_MMI, quantize_decoder))
    else:
        save_dir = "./LUTs/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}_ChannelUniform".format(N, K, quantize_channel_uniform, quantize_decoder)
        print("channel uniform quantization level = {:d}\ndecoder quantization level = {:d}".format(quantize_channel_uniform, quantize_decoder))

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

        if is_quantize_channel_MMI:
            # Channel Quantizer
            print("Generate Channel MMI Quantizer...")
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

        # Generate LUT for every building block of polar codes
        print("Generate SC MMI Quantizer...")
        decoder_quantizer = SCQuantizer(N, K, quantize_decoder, frozenbits)
        lut_fs, lut_gs, virtual_channel_llrs, virtual_channel_probs = decoder_quantizer.find_quantizer(pzx)

        print("LUT Generation Finished...")

        save_path_f = os.path.join(save_dir, "LUT_F_EbN0dB={:d}.pkl".format(EbN0dB))
        save_path_g = os.path.join(save_dir, "LUT_G_EbN0dB={:d}.pkl".format(EbN0dB))
        save_path_virtual_channel_llr = os.path.join(save_dir, "LLR_EbN0dB={:d}.pkl".format(EbN0dB))
        save_path_virtual_channel_probs = os.path.join(save_dir, "Probs_EbN0dB={:d}.pkl".format(EbN0dB))
        with open(save_path_f, "wb") as f:
            pkl.dump(lut_fs, f)
        with open(save_path_g, "wb") as g:
            pkl.dump(lut_gs, g)
        with open(save_path_virtual_channel_llr, "wb") as v:
            pkl.dump(virtual_channel_llrs, v)
        with open(save_path_virtual_channel_probs, "wb") as v:
            pkl.dump(virtual_channel_probs, v)

        print("Save LUTs for f function to : {:s}".format(save_path_f))
        print("Save LUTs for g function to : {:s}".format(save_path_g))
        print("Save virtual channel llr to : {:s}".format(save_path_virtual_channel_llr))
        print("Save virtual channel probs to : {:s}".format(save_path_virtual_channel_probs))

if __name__ == '__main__':
    main()
