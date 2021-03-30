import numpy as np
from utils import continous2discret, channel_transition_probability_table
from PolarDecoder.Decoder.SCLUTDecoder import SCLUTDecoder
from PolarDecoder.Decoder.SCLLUTDecoder import SCLLUTDecoder
from PolarDecoder.Decoder.FastSCLUTDecoder import FastSCLUTDecoder
from PolarDecoder.Decoder.FastSCLLUTDecoder import FastSCLLUTDecoder
from PolarDecoder.Decoder.CASCLLUTDecoder import CASCLLUTDecoder
from quantizers.quantizer.MMI import MMIQuantizer
from PolarBDEnc.Encoder.PolarEnc import PolarEnc
from PolarBDEnc.Encoder.CRCEnc import CRCEnc
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
from ConventionalDecoder.IdentifyNodes import NodeIdentifier

from torchtracer import Tracer
from torchtracer.data import Config

import argparse
import os
import pickle as pkl
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=512)
parser.add_argument("--A", type=int, default=32)
parser.add_argument("--L", type=int, default=8)
parser.add_argument("--DecoderType", type=str, default="CA-SCL")
parser.add_argument("--QuantizationAlgorithm", type=str, default="MMI")
parser.add_argument("--isCRC",type=str, default="yes")
parser.add_argument("--QChannelUniform", type=int, default=128)
parser.add_argument("--QDecoder", type=int, default=16)
parser.add_argument("--QChannel", type=int, default=16)
parser.add_argument("--DesignSNRdB", type=float, default=3.0)
args = parser.parse_args()

# CRC encoding scheme in 5G NR PDCCH polar encoding
isCRC = args.isCRC
crc_n = 24  # CRC check code length
crc_p = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]  # CRC generator polynomial

# polar code parameter
N = args.N  # code length
A = args.A  # information bits amount
if isCRC == "yes":
    K = args.A + crc_n
else:
    K = args.A
L = args.L # list size
rate = A / N # true ode rate
DecoderType = args.DecoderType

if isCRC == "yes" and DecoderType[:2] != "CA":
    raise RuntimeError("You are using CRC added encoding scheme and it seems that you are not using CA-type"
                       " decoding algorithm. Try using --DecoderType CA-SCL")
if DecoderType[-2:] == "SC" and L != 1:
    raise RuntimeError("You are using SC or FastSC decoder and it seems that you erroneously set L != 1. Try setting --L 1")

# quantization parameter
QDecoder = args.QDecoder
QChannel = args.QChannel
QChannelUniform = args.QChannelUniform

load_dir = ""

# code construction
constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction

# node type identification
node_identifier = NodeIdentifier(N, K, frozenbits, msgbits, use_new_node=False)
node_type = node_identifier.run().astype(np.int32)

# initialize encoder and decoder
polar_encoder = PolarEnc(N, K, frozenbits, msgbits)
crc_encoder = CRCEnc(crc_n, crc_p)

DesignEbN0dB = args.DesignSNRdB
load_path_f = os.path.join(load_dir, "LUT_F_EbN0dB={:d}.pkl".format(DesignEbN0dB))
load_path_g = os.path.join(load_dir, "LUT_G_EbN0dB={:d}.pkl".format(DesignEbN0dB))
load_path_virtual_channel_llr = os.path.join(load_dir, "LLR_EbN0dB={:d}.pkl".format(DesignEbN0dB))
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

DecoderDict = {"SC-LUT": SCLUTDecoder(N, K, frozenbits_indicator, messagebits_indicator, fs, gs, virtual_channel_llrs),
               "SCL-LUT": SCLLUTDecoder(N, K, L, frozenbits_indicator, messagebits_indicator, fs, gs, virtual_channel_llrs),
               "FastSC-LUT": FastSCLUTDecoder(N, K, frozenbits_indicator, messagebits_indicator, node_type, fs, gs, virtual_channel_llrs),
               "FastSCL-LUT": FastSCLLUTDecoder(N, K, L, frozenbits_indicator, messagebits_indicator, node_type, fs, gs, virtual_channel_llrs),
               "CA-SCL-LUT": CASCLLUTDecoder(N, K, A, L, frozenbits_indicator, messagebits_indicator, crc_n, crc_p, fs, gs, virtual_channel_llrs)}

polar_decoder = DecoderDict[DecoderType]

# configure experiment tracer
experiment_name = "{:s}-N={:d}-A={:d}-L={:d}-CRC={:s}".format(DecoderType, N, A, L, isCRC)
if os.path.isdir(os.path.join(os.getcwd(), "simulation result", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "simulation result", experiment_name))
tracer = Tracer('simulation result').attach(experiment_name)
configure = {"N": N,
             "A": A,
             "K": K,
             "L": L,
             }
tracer.store(Config(configure))

# simulation parameter configuration
MaxBlock = 10**5
EbN0dBTest = [0, 1, 2, 3, 4, 5]
# BER/FER
ber = []
fer = []
# timing variables
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

    # Build physical channel quantizer under the specific Eb/N0, here we build MMI quantizer for all quantized decoder
    highest = 1 + 3 * sigma
    lowest = -1 - 3 * sigma
    ChannelQuantizer = MMIQuantizer(px1=0.5, px_minus1=0.5)
    pyx1, interval_x = channel_transition_probability_table(QChannelUniform, lowest, highest, 1, sigma)
    pyx_minus1, _ = channel_transition_probability_table(QChannelUniform, lowest, highest, -1, sigma)
    joint_prob = np.zeros((2, QChannelUniform)).astype(np.float32)
    joint_prob[0] = pyx1
    joint_prob[1] = pyx_minus1
    channel_lut = ChannelQuantizer.find_opt_quantizer_AWGN(joint_prob, QChannel)
    pzx = np.zeros((2, int(QChannel)))
    for i in range(int(QChannel)):
        begin = channel_lut[i]
        end = channel_lut[i + 1]
        pzx[0, i] = np.sum(pyx1[begin:end])
        pzx[1, i] = np.sum(pyx_minus1[begin:end])

    Nbiterrs = 0
    Nblkerrs = 0
    Nblocks = 0
    pbar = tqdm(range(MaxBlock))
    for _ in pbar:

        msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for valina SC

        cword = polar_encoder.encode(msg)

        bpsksymbols = 1 - 2 * cword  # BPSK modulation

        y = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

        y_symbols = np.zeros(N)

        for i in range(N):
            y_symbols[i] = continous2discret(y[0, i], interval_x[channel_lut], QChannel - 1)

        decoded_bits = polar_decoder.decode(y_symbols)

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
            # calc BER and FER in the given EbN0
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


