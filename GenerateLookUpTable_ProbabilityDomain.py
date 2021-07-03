import numpy as np
from QuantizeDensityEvolution.QDensityEvolution_MMI import QDensityEvolutionMMI
from QuantizeDensityEvolution.QLLRDensityEvolution_MsIB import QDensityEvolutionMsIB
from BitChannelMerge.DegradingMerge import DegradeMerger
from quantizers.quantizer.MMI import MMIQuantizer
from utils import channel_transition_probability_table
import pickle as pkl
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--QDecoder", type=int, default=16)
    parser.add_argument("--QChannelUniform", type=int, default=128)
    parser.add_argument("--QChannelCompressed", type=int, default=16)
    parser.add_argument("--DesignSNRdB", type=float, default=3.0)
    parser.add_argument("--Quantizer", type=str, default="DegradeMerge")
    args = parser.parse_args()

    N = args.N  # code length
    QDecoder = args.QDecoder
    QChannelCompressed = args.QChannelCompressed
    QChannelUniform = args.QChannelUniform
    DesignSNRdB = args.DesignSNRdB
    Quantizer = args.Quantizer

    ChannelQuantizerDict = {"MMI": MMIQuantizer(px1=0.5, px_minus1=0.5)}
    DecoderQuantizerDict = {"MMI": QDensityEvolutionMMI(N, QDecoder),
                            "MsIB": QDensityEvolutionMsIB(N, QDecoder),
                            "DegradeMerge": DegradeMerger(N, QDecoder)}

    save_dir = "./LUT/{:s}/N{:d}_ChannelQ{:d}_DecoderQ{:d}".format(Quantizer, N, QChannelCompressed, QDecoder)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # simulation parameter configuration
    DesignSNR = 10**(DesignSNRdB/10)  # linear scale snr
    sigma = np.sqrt(1/DesignSNR)  # Gaussian noise variance for current EbN0

    # Channel Quantizer
    highest = 1 + 3 * sigma
    lowest = -1 - 3 * sigma
    ChannelQuantizer = ChannelQuantizerDict["MMI"]
    pyx1, interval_x = channel_transition_probability_table(QChannelUniform, lowest, highest, 1, sigma)
    pyx_minus1, _ = channel_transition_probability_table(QChannelUniform, lowest, highest, -1, sigma)
    joint_prob = np.zeros((2, QChannelUniform)).astype(np.float32)
    joint_prob[0] = pyx1
    joint_prob[1] = pyx_minus1
    channel_lut = ChannelQuantizer.find_opt_quantizer_AWGN(joint_prob, QChannelCompressed)
    pzx = np.zeros((2, int(QChannelCompressed)))
    for i in range(int(QChannelCompressed)):
        begin = channel_lut[i]
        end = channel_lut[i + 1]
        pzx[0, i] = np.sum(pyx1[begin:end])
        pzx[1, i] = np.sum(pyx_minus1[begin:end])

    # Generate LUT for every building block of polar codes
    decoder_quantizer = DecoderQuantizerDict[Quantizer]
    lut_fs, lut_gs, virtual_channel_llrs, virtual_channel_probs = decoder_quantizer.run(pzx)

    save_path_f = os.path.join(save_dir, "LUT_F_SNRdB={:.1f}.pkl".format(DesignSNRdB))
    save_path_g = os.path.join(save_dir, "LUT_G_SNRdB={:.1f}.pkl".format(DesignSNRdB))
    save_path_virtual_channel_llr = os.path.join(save_dir, "LLR_SNRdB={:.1f}.pkl".format(DesignSNRdB))
    with open(save_path_f, "wb") as f:
        pkl.dump(lut_fs, f)
    with open(save_path_g, "wb") as g:
        pkl.dump(lut_gs, g)
    with open(save_path_virtual_channel_llr, "wb") as v:
        pkl.dump(virtual_channel_llrs, v)

    print("Save LUTs for f function to : {:s}".format(save_path_f))
    print("Save LUTs for g function to : {:s}".format(save_path_g))
    print("Save virtual channel llr to : {:s}".format(save_path_virtual_channel_llr))

if __name__ == '__main__':
    main()