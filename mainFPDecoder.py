'''
This file is the entrance for float point decoders for Polar Codes.
'''
import numpy as np
from PolarDecoder.Decoder.FastSCDecoder import FastSCDecoder
from PolarDecoder.Decoder.SCDecoder import SCDecoder
from PolarDecoder.Decoder.SCLDecoder import SCLDecoder
from PolarDecoder.Decoder.FastSCLDecoder import FastSCLDecoder
from PolarDecoder.Decoder.CASCLDecoder import CASCLDecoder
from PolarCodesUtils.IdentifyNodes import NodeIdentifier
from PolarCodesUtils.CodeConstruction import PolarCodeConstructor
from PolarBDEnc.Encoder.PolarEnc import PolarEnc
from PolarBDEnc.Encoder.CRCEnc import CRCEnc
import argparse
from tqdm import tqdm
import os
from torchtracer import Tracer
from torchtracer.data import Config
import shutil
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=512)
parser.add_argument("--A", type=int, default=32)
parser.add_argument("--L", type=int, default=8)
parser.add_argument("--DecoderType", type=str, default="CA-SCL")
parser.add_argument("--isCRC",type=str, default="yes")
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

# code constructor
constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction

# node type identification
node_identifier = NodeIdentifier(N, K, frozenbits, msgbits, use_new_node=False)
node_type = node_identifier.run().astype(np.int32)

# Polar encoder and decoder initialization
polar_encoder = PolarEnc(N, K, frozenbits, msgbits)
crc_encoder = CRCEnc(crc_n, crc_p)
DecoderDict = {"SC": SCDecoder(N, K, frozenbits_indicator, messagebits_indicator),
               "SCL": SCLDecoder(N, K, L, frozenbits_indicator, messagebits_indicator),
               "FastSC": FastSCDecoder(N, K, frozenbits_indicator, messagebits_indicator, node_type),
               "FastSCL": FastSCLDecoder(N, K, L, frozenbits_indicator, messagebits_indicator, node_type),
               "CA-SCL": CASCLDecoder(N, K, A, L, frozenbits_indicator, messagebits_indicator, crc_n, crc_p)}

polar_decoder = DecoderDict[DecoderType]

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
EbN0dBtest = [0, 1, 2, 3, 4]
BER = []
BLER = []
total_blocks = 0
print("---------Summary---------")
print("Decoder : {:s}".format(DecoderType))
print("Code Length (N) = {:d}".format(N))
print("# Information Bits (A) = {:d}".format(A))
print("List Size (L) = {:d}".format(L))
print("Using CRC? {:s}".format(isCRC))
print("CRC Length = {:d}".format(crc_n))
print("-------------------------")
# start simulation
for EbN0dB in EbN0dBtest:
    EbN0 = 10**(EbN0dB/10)
    sigma = np.sqrt(1/(2*rate*EbN0))
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

        bpsksymbols = 1 - 2 * cword  # BPSK modulation

        receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

        receive_symbols_llr = receive_symbols * (2/sigma**2) # symbol -> LLR

        decoded_bits = polar_decoder.decode(receive_symbols_llr)

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

# save BER/BLER curve

plt.figure(dpi=300)
plt.semilogy(EbN0dBtest, BER, color='r', linestyle='-', marker="*", markersize=5)
plt.legend(["{:s}".format(DecoderType)])
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid()
tracer.store(plt.gcf(), "BER.png")

plt.figure(dpi=300)
plt.semilogy(EbN0dBtest, BLER, color='r', linestyle='-', marker="*", markersize=5)
plt.legend(["{:s}".format(DecoderType)])
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Block Error Rate (BLER)")
plt.grid()
tracer.store(plt.gcf(), "BLER.png")
