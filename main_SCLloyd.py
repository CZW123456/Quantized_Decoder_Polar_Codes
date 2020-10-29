import numpy as np
from PolarDecoder.Decoder.SCLloydQuantizedDecoder import SCLloydQuantizedDecoder
from QuantizeDecoder.LloydQuantizer import LloydQuantizer
from QuantizeDecoder.LLRLloydSC import LLRLloydGA
from ConventionalDecoder.encoder import PolarEncoder
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import argparse
import os
from tqdm import tqdm
from bisect import bisect_left
import pickle as pkl


def llr(inp, sigma):
    return (2/sigma ** 2) * inp


def Q(llrs, boundary, reconstruct):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--v", type=int, default=16)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    v = args.v  # quantization level
    rate = K / N

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozen_indicator, message_indicator = constructor.PW()  # PW code construction
    print("N = {:d}, K = {:d}, R = {:.2f}, v = {:d}".format(N, K, rate, v))


    # initialize encoder and decoder
    polar_encoder = PolarEncoder(N, K, frozenbits, msgbits)
    Q_channel = LloydQuantizer(v, max_iter=1000)
    Q_decoder = LLRLloydGA(N, v, max_iter=1000)
    # simulation parameter configuration
    MaxBlock = 10**6
    EbN0dBTest = [0, 1, 2, 3, 4, 5]
    # BER/FER
    ber = []
    fer = []
    # design quantized decoder in a given Eb/N0
    design_SNR = 3
    save_dir = "./LUTs_Lloyd/{:d}dB".format(design_SNR)
    if os.path.isfile(os.path.join(save_dir, "N{:d}_K{:d}_v{:d}".format(N, K, v))):
        with open(os.path.join(save_dir, "N{:d}_K{:d}_v{:d}".format(N, K, v)), "rb") as f:
            decoder_boundary_f, decoder_boundary_g, decoder_reconstruct_f, decoder_reconstruct_g = pkl.load(f)
    else:
        EbN0 = 10 ** (design_SNR / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0
        decoder_boundary_f, decoder_boundary_g, decoder_reconstruct_f, decoder_reconstruct_g = Q_decoder.generate_Lloyd_quantizers(
            sigma)
        decoder_boundary_f = decoder_boundary_f.tolist()
        decoder_boundary_g = decoder_boundary_g.tolist()
        decoder_reconstruct_f = decoder_reconstruct_f.tolist()
        decoder_reconstruct_g = decoder_reconstruct_g.tolist()
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "N{:d}_K{:d}_v{:d}".format(N, K, v)), "wb") as f:
            pkl.dump([decoder_boundary_f, decoder_boundary_g, decoder_reconstruct_f, decoder_reconstruct_g], f)
    polar_decoder = SCLloydQuantizedDecoder(N, K, frozen_indicator, message_indicator, decoder_boundary_f, decoder_boundary_g,
                                            decoder_reconstruct_f, decoder_reconstruct_g, v)
    total_blocks = 0
    for EbN0dB in EbN0dBTest:
        EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
        sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0
        # generate uniform quantized decoder
        mu_llr = 2/sigma**2
        sigma2_llr = 2*mu_llr
        boundary_channel, reconstruct_channel = Q_channel.find_quantizer_gaussian(mu_llr, sigma2_llr,
                                                                                  begin=-mu_llr-3*np.sqrt(sigma2_llr),
                                                                                  end=mu_llr+3*np.sqrt(sigma2_llr))
        Nbiterrs = 0
        Nblkerrs = 0
        Nblocks = 0

        pbar = tqdm(range(MaxBlock))
        for _ in pbar:

            msg = np.random.randint(low=0, high=2, size=K)  # generate 0-1 msg bits for valina SC

            cword = polar_encoder.non_system_encoding(msg)

            bpsksymbols = 1 - 2 * cword  # BPSK modulation

            receive_symbols = bpsksymbols + np.random.normal(loc=0, scale=sigma, size=(1, N))  # AWGN noisy channel

            receive_symbols_llr = llr(receive_symbols, sigma)  # calc channel llr
            quantized_llr = Q(receive_symbols_llr, boundary_channel, reconstruct_channel)
            decoded_bits = polar_decoder.decode(quantized_llr)  # valina SC Decoder

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

    # save BER/BLER
    ber_save_dir = "simulation_result/BER/SCLloyd"
    bler_save_dir = "simulation_result/BLER/SCLloyd"
    if not os.path.isdir(ber_save_dir):
        os.makedirs(ber_save_dir)
    if not os.path.isdir(bler_save_dir):
        os.makedirs(bler_save_dir)
    ber_save_path = os.path.join(ber_save_dir, "N{:d}_K{:d}_v{:d}".format(N, K, v))
    bler_save_path = os.path.join(bler_save_dir, "N{:d}_K{:d}_v{:d}".format(N, K, v))
    ber = np.array(ber)
    bler = np.array(fer)
    np.savetxt(ber_save_path, ber, fmt='%.8f', delimiter='\n')
    np.savetxt(bler_save_path, bler, fmt='%.8f', delimiter='\n')


if __name__ == '__main__':
    main()
