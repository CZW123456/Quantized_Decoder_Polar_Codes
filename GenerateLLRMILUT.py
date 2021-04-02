import numpy as np
from QuantizeDecoder.LLRMIQuantizedSC import LLRMIQuantizerSC
from QuantizeDecoder.LLRMIQuantizer import LLRMIQuantizer
from PolarCodesUtils.CodeConstruction import PolarCodeConstructor
import pickle as pkl
import argparse
import os

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

    save_dir = "./LUTs_LLRMI/N{:d}_K{:d}_ChannelQ{:d}_DecoderQ{:d}".format(N, K, quantize_channel, quantize_decoder)
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
        sigma_awgn = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0

        print("Eb/N0 = {:0f} dB, Sigma2 = {:6f}".format(EbN0dB, sigma_awgn))

        # Channel Quantizer
        mu = 2 / sigma_awgn ** 2
        sigma2_llr = 2 * mu
        begin = -mu - 3 * np.sqrt(sigma2_llr)
        end = mu + 3 * np.sqrt(sigma2_llr)
        quantizer = LLRMIQuantizer(quantize_channel)
        boundary, reconstruction, plx = quantizer.find_quantizer_awgn(mu, sigma2_llr, begin, end)

        # Generate LUT for every building block of polar codes
        print("Generate SC MMI Quantizer...")
        SCQuantizer = LLRMIQuantizerSC(N, quantize_decoder, quantize_decoder)
        llr_density, llr_quanta, lut_fs, lut_gs = SCQuantizer.generate_LUTs(channel_llr_density=plx,
                                                                            channel_llr_quanta=reconstruction)
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