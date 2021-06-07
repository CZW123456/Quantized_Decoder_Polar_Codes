import numpy as np
from utils import channel_transition_probability_table
from quantizers.quantizer.MMI import MMIQuantizer
from QuantizeDecoder.LLRQuantizedSC import LLRQuantizerSC
import pickle as pkl
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--QDecoder", type=int, default=16)
    parser.add_argument("--QChannelUniform", type=int, default=128)
    parser.add_argument("--QChannelCompressed", type=int, default=16)
    parser.add_argument("--DesignSNRdB", type=float, default=3.0)
    parser.add_argument("--Quantizer", type=str, default="MinDistortion")

    args = parser.parse_args()

    # polar codes configuration
    N = args.N
    QDecoder = args.QDecoder
    QChannelCompressed = args.QChannelCompressed
    QChannelUniform = args.QChannelUniform
    DesignSNRdB = args.DesignSNRdB
    Quantizer = args.Quantizer

    ChannelQuantizerDict = {"MMI": MMIQuantizer(px1=0.5, px_minus1=0.5)}
    DecoderQuantizerDict = {"MinDistortion": LLRQuantizerSC(N, QDecoder)}

    save_dir = "./LUT/{:s}/N{:d}_ChannelQ{:d}_DecoderQ{:d}".format(Quantizer, N, QChannelCompressed, QDecoder)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # simulation parameter configuration
    DesignSNR = 10 ** (DesignSNRdB / 10)  # linear scale snr
    sigma = np.sqrt(1 / DesignSNR)  # Gaussian noise variance for current EbN0

    print("SNR = {:.1f} dB, Sigma2 = {:.3f}".format(DesignSNRdB, sigma))

    # Channel Quantizer
    E_LLR = 2 / (sigma ** 2)
    V_LLR = 2 * E_LLR
    D_LLR = np.sqrt(V_LLR)
    highest = E_LLR + 3 * D_LLR
    lowest = -E_LLR - 3 * D_LLR
    ChannelQuantizer = ChannelQuantizerDict["MMI"]
    pyx1, interval_x, = channel_transition_probability_table(QChannelUniform, lowest, highest, E_LLR, D_LLR)
    pyx_minus1, _, = channel_transition_probability_table(QChannelUniform, lowest, highest, -E_LLR, D_LLR)
    quanta_channel_uniform = interval_x[:-1] + 0.5 * (interval_x[1] - interval_x[0])
    joint_prob = np.zeros((2, QChannelUniform)).astype(np.float32)
    joint_prob[0] = pyx1
    joint_prob[1] = pyx_minus1
    channel_lut = ChannelQuantizer.find_opt_quantizer_AWGN(joint_prob, QChannelCompressed)
    quanta_channel = np.zeros(QChannelCompressed)
    channel_llr_density = np.zeros(QChannelCompressed)
    for i in range(int(QChannelCompressed)):
        begin = channel_lut[i]
        end = channel_lut[i + 1]
        pxz = 0.5 * (pyx1[begin:end] + pyx_minus1[begin:end])
        quanta_channel[i] = np.sum(quanta_channel_uniform[begin:end] * pxz) / np.sum(pxz)
        channel_llr_density[i] = np.sum(pxz)

    # Generate LUT for every building block of polar codes
    print("LUT Generation Begins...")
    decoder_quantizer = DecoderQuantizerDict[Quantizer]
    llr_density, llr_quanta, lut_fs, lut_gs = decoder_quantizer.run(channel_llr_density=channel_llr_density,
                                                                    channel_llr_quanta=quanta_channel)
    print("LUT Generation Finished...")

    save_path_f = os.path.join(save_dir, "LUT_F_SNRdB={:.0f}.pkl".format(DesignSNRdB))
    save_path_g = os.path.join(save_dir, "LUT_G_SNRdB={:.0f}.pkl".format(DesignSNRdB))
    save_path_llr_quanta = os.path.join(save_dir, "LLRQuanta_SNRdB={:.0f}.pkl".format(DesignSNRdB))
    save_path_llr_density = os.path.join(save_dir, "LLRDensity_SNRdB={:.0f}.pkl".format(DesignSNRdB))
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