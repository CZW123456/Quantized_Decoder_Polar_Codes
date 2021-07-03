import numpy as np

from PolarDecoder.Decoder.SCLUniformQuantizedDecoder import SCLUniformQuantizedDecoder
from PolarDecoder.Decoder.SCUniformQuantizedDecoder import SCUniformQuantizedDecoder
from PolarDecoder.Decoder.SCLLloydQuantizedDecoder import SCLLloydQuantizedDecoder
from PolarDecoder.Decoder.SCLloydQuantizedDecoder import SCLloydQuantizedDecoder

from QuantizeDensityEvolution.OptUniformQuantizerGaussian import OptUniformQuantizerGaussian
from QuantizeDensityEvolution.QLLRDensityEvolution_OptUniform import LLRLSUniformQuantizer
from QuantizeDensityEvolution.LloydQuantizer import LloydQuantizer
from QuantizeDensityEvolution.QLLRDensityEvolution_Lloyd import LLRLloydGA

from PolarBDEnc.Encoder.PolarEnc import PolarEnc
from PolarBDEnc.Encoder.CRCEnc import CRCEnc
from PolarCodesUtils.CodeConstruction import PolarCodeConstructor

from bisect import bisect_left

from torchtracer import Tracer
from torchtracer.data import Config
import argparse
import os
import shutil
from tqdm import tqdm


def QUniform(llr, M, r):
    llr[np.abs(llr) <= M] = (np.floor(llr[np.abs(llr) <= M] / r).astype(int) + 1 / 2) * r
    llr[np.abs(llr) > M] = np.sign(llr[np.abs(llr) > M]) * (M - r / 2)
    return llr

def QLloyd(llrs, boundary, reconstruct):
    quantized_result = []
    for llr in llrs[0]:
        if llr <= boundary[0]:
            return reconstruct[0]
        if llr >= boundary[-1]:
            return reconstruct[-1]
        i = int(bisect_left(boundary, llr))
        quantized_result.append(reconstruct[i - 1])
    quantized_result = np.asarray(quantized_result)
    return quantized_result


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=512)
parser.add_argument("--A", type=int, default=32)
parser.add_argument("--L", type=int, default=8)
parser.add_argument("--DecoderType", type=str, default="SC-Lloyd")
parser.add_argument("--QuantizationAlgorithm", type=str, default="Uniform")
parser.add_argument("--isCRC",type=str, default="no")
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
QuantizationAlgorithm = args.QuantizationAlgorithm

if isCRC == "yes" and DecoderType[:2] != "CA":
    raise RuntimeError("You are using CRC added encoding scheme and it seems that you are not using CA-type"
                       " decoding algorithm. Try using --DecoderType CA-SCL")
if DecoderType[-2:] == "SC" and L != 1:
    raise RuntimeError("You are using SC or FastSC decoder and it seems that you erroneously set L != 1. Try setting --L 1")

# quantization parameter
QDecoder = args.QDecoder
QChannel = args.QChannel
QChannelUniform = args.QChannelUniform

# code construction
constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction

# initialize encoder and decoder
polar_encoder = PolarEnc(N, K, frozenbits, msgbits)
crc_encoder = CRCEnc(crc_n, crc_p)

# initialize uniform/Lloyd quantizer for AWGN channels and internal LLRs
UniformChannelQuantizer = OptUniformQuantizerGaussian(QChannel)
LloydChannelQuantizer = LloydQuantizer(QDecoder, max_iter=200)
QDecoderUniform = LLRLSUniformQuantizer(N, QChannel)
QDecoderLloyd = LLRLloydGA(N, QDecoder, max_iter=200)
DesignEbN0dB = args.DesignSNRdB
EbN0 = 10**(DesignEbN0dB/10)  # linear scale snr
sigma = np.sqrt(1/(2*rate*EbN0))  # Gaussian noise variance for current EbN0

if QuantizationAlgorithm == "Uniform":
    decoder_r_f, decoder_r_g = QDecoderUniform.generate_uniform_quantizers(sigma)
    decoder_r_f = decoder_r_f.tolist()
    decoder_r_g = decoder_r_g.tolist()
    if DecoderType.split("-")[0] == "SC":
        polar_decoder = SCUniformQuantizedDecoder(N, K, frozenbits_indicator, messagebits_indicator, decoder_r_f, decoder_r_g, QDecoder)
    elif DecoderType.split("-")[0] == "SCL":
        SCLUniformQuantizedDecoder(N, K, L, frozenbits_indicator, messagebits_indicator, decoder_r_f, decoder_r_g, QDecoder)
    else:
        raise RuntimeError("DecoderType should begin with either SC or SCL")

elif QuantizationAlgorithm == "Lloyd":
    decoder_boundary_f, decoder_boundary_g, decoder_reconstruct_f, decoder_reconstruct_g = QDecoderLloyd.generate_Lloyd_quantizers(sigma)
    decoder_boundary_f = decoder_boundary_f.tolist()
    decoder_boundary_g = decoder_boundary_g.tolist()
    decoder_reconstruct_f = decoder_reconstruct_f.tolist()
    decoder_reconstruct_g = decoder_reconstruct_g.tolist()
    if DecoderType.split("-")[0] == "SC":
        polar_decoder = SCLloydQuantizedDecoder(N, K, frozenbits_indicator, messagebits_indicator, decoder_boundary_f,
                                                decoder_boundary_g, decoder_reconstruct_f, decoder_reconstruct_g, QDecoder)
    elif DecoderType.split("-")[0] == "SCL":
        polar_decoder = SCLLloydQuantizedDecoder(N, K, L, frozenbits_indicator, messagebits_indicator, decoder_boundary_f,
                                                 decoder_boundary_g, decoder_reconstruct_f, decoder_reconstruct_g,
                                                 QDecoder)
    else:
        raise RuntimeError("DecoderType should begin with either SC or SCL")
else:
    raise RuntimeError("Quantization method for AWGN channel should be either Uniform or Lloyd")

QFuncDict = {"Uniform":QUniform, "Lloyd":QLloyd}
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

    # Build physical channel quantizer for LLR soft values under the specific Eb/N0
    mu_llr = 2 / sigma ** 2
    sigma2_llr = 2 * mu_llr

    if QuantizationAlgorithm == "Uniform":
        reconstruct_channel = UniformChannelQuantizer.find_optimal_interval_bimodal_Gaussian(mu_llr, sigma2_llr, 30)
        boundary_channel = (QChannel // 2 - 1) * reconstruct_channel
    elif QuantizationAlgorithm == "Lloyd":
        boundary_channel, reconstruct_channel = LloydChannelQuantizer.find_quantizer_gaussian(mu_llr, sigma2_llr,
                                                                                              begin=-mu_llr - 3*np.sqrt(sigma2_llr),
                                                                                              end=mu_llr + 3*np.sqrt(sigma2_llr))
    else:
        raise RuntimeError("Quantization method for AWGN channel should be either Uniform or Lloyd")
    q = QFuncDict[QuantizationAlgorithm]
    Nbiterrs = 0
    Nblkerrs = 0
    Nblocks = 0
    pbar = tqdm(range(MaxBlock))
    for _ in pbar:

        msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for valina SC

        cword = polar_encoder.encode(msg)

        bpsksymbols = 1 - 2 * cword  # BPSK modulation

        y = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

        llr = y * 2/(sigma**2) # convert noisy symbol soft value to LLR

        qllr = q(llr, boundary_channel, reconstruct_channel)

        decoded_bits = polar_decoder.decode(qllr)

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

