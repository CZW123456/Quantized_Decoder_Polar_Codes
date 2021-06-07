import numpy as np
from bisect import bisect_left
from utils import channel_llr_density_table
from PolarDecoder.Decoder.SCLUTDecoder import SCLUTDecoder
from PolarDecoder.Decoder.SCLLUTDecoder import SCLLUTDecoder
from PolarDecoder.Decoder.FastSCLUTDecoder import FastSCLUTDecoder
from PolarDecoder.Decoder.FastSCLLUTDecoder import FastSCLLUTDecoder
from PolarDecoder.Decoder.CASCLLUTDecoder import CASCLLUTDecoder
from quantizers.quantizer.LLROptLSQuantizer import LLRQuantizer
from PolarBDEnc.Encoder.PolarEnc import PolarEnc
from PolarBDEnc.Encoder.CRCEnc import CRCEnc
from PolarCodesUtils.CodeConstruction import PolarCodeConstructor
from PolarCodesUtils.IdentifyNodes import NodeIdentifier

from torchtracer import Tracer
from torchtracer.data import Config

import argparse
import os
import pickle as pkl
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=256)
parser.add_argument("--A", type=int, default=32)
parser.add_argument("--L", type=int, default=8)
parser.add_argument("--DecoderType", type=str, default="SC-LUT")
parser.add_argument("--Quantizer", type=str, default="MinDistortion")
parser.add_argument("--isCRC",type=str, default="no")
parser.add_argument("--QChannelUniform", type=int, default=128)
parser.add_argument("--QChannelCompressed", type=int, default=16)
parser.add_argument("--QDecoder", type=int, default=16)
parser.add_argument("--DesignSNRdB", type=float, default=3.0)
args = parser.parse_args()

# CRC encoding scheme in 5G NR PDCCH polar encoding
isCRC = args.isCRC
crc_n = 24  # CRC check code length
crc_p = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]  # CRC generator polynomial

# polar code parameter
N = args.N
A = args.A
if isCRC == "yes":
    K = args.A + crc_n
else:
    K = args.A
L = args.L
rate = A / N # effective code rate
DecoderType = args.DecoderType

# quantization parameter
QDecoder = args.QDecoder
QChannelCompressed = args.QChannelCompressed
QChannelUniform = args.QChannelUniform
Quantizer = args.Quantizer

load_dir = "./LUT/{:s}/N{:d}_ChannelQ{:d}_DecoderQ{:d}".format(Quantizer, N, QChannelCompressed, QDecoder)

# code construction
constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction

# node type identification
node_identifier = NodeIdentifier(N, K, frozenbits, msgbits, use_new_node=False)
node_type = node_identifier.run().astype(np.int32)

# initialize encoder and decoder
polar_encoder = PolarEnc(N, K, frozenbits, msgbits)
crc_encoder = CRCEnc(crc_n, crc_p)

DesignSNRdB = args.DesignSNRdB
load_path_f = os.path.join(load_dir, "LUT_F_SNRdB={:.0f}.pkl".format(DesignSNRdB))
load_path_g = os.path.join(load_dir, "LUT_G_SNRdB={:.0f}.pkl".format(DesignSNRdB))
load_path_virtual_channel_llr = os.path.join(load_dir, "LLRQuanta_SNRdB={:.0f}.pkl".format(DesignSNRdB))
with open(load_path_f, "rb+") as f:
    lut_fs = pkl.load(f)
with open(load_path_g, "rb+") as f:
    lut_gs = pkl.load(f)
with open(load_path_virtual_channel_llr, "rb") as f:
    virtual_channel_llrs = pkl.load(f)

# load quantized decoder
virtual_channel_llrs = virtual_channel_llrs.tolist()
lut_fs = np.array([np.array(lut_fs[key]).astype(np.int32) for key in lut_fs.keys()])
lut_gs = np.array([np.array(lut_gs[key]).astype(np.int32) for key in lut_gs.keys()])
fs = []
for ele in lut_fs:
    fs.append(ele.tolist())
gs = []
for ele in lut_gs:
    gs.append(ele.tolist())


ChannelQuantizerDict = {"MinDistortion":LLRQuantizer()}
DecoderDict = {"SC-LUT": SCLUTDecoder(N, K, frozenbits_indicator, messagebits_indicator, fs, gs, virtual_channel_llrs),
               "SCL-LUT": SCLLUTDecoder(N, K, L, frozenbits_indicator, messagebits_indicator, fs, gs, virtual_channel_llrs),
               "FastSC-LUT": FastSCLUTDecoder(N, K, frozenbits_indicator, messagebits_indicator, node_type, fs, gs, virtual_channel_llrs),
               "FastSCL-LUT": FastSCLLUTDecoder(N, K, L, frozenbits_indicator, messagebits_indicator, node_type, fs, gs, virtual_channel_llrs),
               "CASCL-LUT": CASCLLUTDecoder(N, K, A, L, frozenbits_indicator, messagebits_indicator, crc_n, crc_p, fs, gs, virtual_channel_llrs)}

polar_decoder = DecoderDict[DecoderType]

# configure experiment tracer
experiment_name = "{:s}-N={:d}-A={:d}-L={:d}-CRC={:s}".format(DecoderType, N, A, L, isCRC)
if os.path.isdir(os.path.join(os.getcwd(), "simulation result", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "simulation result", experiment_name))
tracer = Tracer('simulation result').attach(experiment_name)
configure = {"N": N, "A": A, "K": K, "L": L}
tracer.store(Config(configure))

# simulation parameter configuration
MaxBlock = 10**5
EbN0dBTest = [0, 1, 2, 3, 4, 5]
total_blocks = 0
BER = []
BLER = []
print("---------Summary---------")
print("Decoder : {:s}".format(DecoderType))
print("Code Length (N) = {:d}".format(N))
print("Information Bits (A) = {:d}".format(A))
print("List Size (L) = {:d}".format(L))
print("Using CRC? {:s}".format(isCRC))
print("CRC Length = {:d}".format(crc_n))
print("-------------------------")
print("Simulation Begin...")
for EbN0dB in EbN0dBTest:

    EbN0 = 10**(EbN0dB/10)
    sigma = np.sqrt(1/(2*rate*EbN0))

    # Build physical channel quantizer for LLR soft values under the specific Eb/N0, here we build MMI quantizer for all quantized decoder
    E_LLR = 2 / (sigma ** 2)
    V_LLR = 2 * E_LLR
    D_LLR = np.sqrt(V_LLR)
    highest = E_LLR + 3 * D_LLR
    lowest = -E_LLR - 3 * D_LLR
    ChannelQuantizer = ChannelQuantizerDict[Quantizer]

    pyx, interval_x, quanta = channel_llr_density_table(QChannelUniform, lowest, highest, E_LLR, -E_LLR, D_LLR)
    channel_llr_density, channel_llr_quanta, channel_lut, _ = ChannelQuantizer.find_OptLS_quantizer(pyx, quanta, QChannelUniform, QChannelCompressed)
    channel_lut = channel_lut.squeeze()

    Nbiterrs = 0
    Nblkerrs = 0
    Nblocks = 0
    pbar = tqdm(range(MaxBlock))
    for _ in pbar:

        msg = np.random.randint(low=0, high=2, size=A)

        if isCRC == "yes":
            msg_crc = crc_encoder.encode(msg)
            cword = polar_encoder.encode(msg_crc).astype(np.int)
        else:
            cword = polar_encoder.encode(msg).astype(np.int)

        bpsksymbols = 1 - 2 * cword

        y = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))

        llr = y * 2 / (sigma ** 2)

        llr_symbols = np.zeros(N).astype(np.int32)

        for i in range(N):
            if llr[0, i] <= interval_x[0]:
                llr_symbols[i] = 0
            elif llr[0, i] >= interval_x[-1]:
                llr_symbols[i] = QChannelCompressed - 1
            else:
                index = bisect_left(interval_x[:-1], llr[0, i])
                llr_symbols[i] = channel_lut[index - 1]

        decoded_bits = polar_decoder.decode(llr_symbols)

        # calc error statistics
        Nbiterrs += np.sum(msg != decoded_bits)
        Nblkerrs += np.any(msg != decoded_bits)

        if Nblkerrs > 1000:
            BER_sim = Nbiterrs / (A * Nblocks)
            BLER_sim = Nblkerrs / Nblocks
            print("EbN0(dB):{:.1f}, BER:{:f}, BLER:{:f}".format(EbN0dB, BER_sim, BLER_sim))
            BER.append(BER_sim)
            BLER.append(BLER_sim)
            tracer.log("{:.6f}".format(BER_sim), file="BER")
            tracer.log("{:.6f}".format(BLER_sim), file="BLER")
            break
        Nblocks += 1
        pbar.set_description("# Error Bits = {:d}, # Error Frame = {:d}".format(Nbiterrs, Nblkerrs))
        if Nblocks == MaxBlock:
            BER_sim = Nbiterrs / (K * Nblocks)
            BLER_sim = Nblkerrs / Nblocks
            print("EbN0(dB):{:.1f}, BER:{:f}, BLER:{:f}".format(EbN0dB, BER_sim, BLER_sim))
            BER.append(BER_sim)
            BLER.append(BLER_sim)
            tracer.log("{:.6f}".format(BER_sim), file="BER")
            tracer.log("{:.6f}".format(BLER_sim), file="BLER")
    total_blocks += Nblocks

plt.figure(dpi=300)
plt.semilogy(EbN0dBTest, BER, color='r', linestyle='-', marker="*", markersize=5)
plt.legend(["{:s}".format(DecoderType)])
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid()
tracer.store(plt.gcf(), "BER.png")

plt.figure(dpi=300)
plt.semilogy(EbN0dBTest, BLER, color='r', linestyle='-', marker="*", markersize=5)
plt.legend(["{:s}".format(DecoderType)])
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Block Error Rate (BLER)")
plt.grid()
tracer.store(plt.gcf(), "BLER.png")