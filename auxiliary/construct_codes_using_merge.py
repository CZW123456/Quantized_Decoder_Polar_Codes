import numpy as np
from BitChannelMerge.utils import get_AWGN_transition_probability_upgrading, get_AWGN_transition_probability_degrading
from BitChannelMerge.UpgradingMerge import UpgradeMerger
from BitChannelMerge.DegradingMerge import DegradeMerger
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N = 64
    K = 32
    rate = K / N
    quantize_level_channel_uniform = 128
    mu = 16
    EbN0dB = 0
    EbN0 = 10 ** (EbN0dB / 10)  # linear scale snr
    sigma = np.sqrt(1 / (2 * rate * EbN0))  # Gaussian noise variance for current EbN0
    print("Eb/N0 = {:0f} dB, Sigma2 = {:6f}".format(EbN0dB, sigma))

    W_AWGN, _ = get_AWGN_transition_probability_upgrading(sigma, mu // 2)
    Merger = UpgradeMerger(mu, N, W_AWGN)
    _, Pe_upgrade, _, _ = Merger.bit_channel_upgrading_run()

    W_AWGN, _ = get_AWGN_transition_probability_degrading(sigma, mu // 2)
    Merger = DegradeMerger(mu, N, W_AWGN)
    _, Pe_degrade, = Merger.bit_channel_degrading_run()

    plt.figure()
    plt.bar(np.arange(N), -np.log10(Pe_degrade))
    plt.xlabel('bit index')
    plt.ylabel(r'$-{\log_{10}P_e}$')
    plt.title('Bit channel error rate estimated under degrade method ($\mu={:d}$)'.format(mu))
    plt.savefig('code construction——degrade.png')

    plt.figure()
    plt.bar(np.arange(N), -np.log10(Pe_upgrade))
    plt.xlabel('bit index')
    plt.ylabel(r'$-{\log_{10}P_e}$')
    plt.title('Bit channel error rate estimated under upgrade method ($\mu={:d}$)'.format(mu))
    plt.savefig('code construction——upgrade.png')

    plt.figure()
    plt.plot(np.arange(N), -np.log10(Pe_degrade), '-k*')
    plt.plot(np.arange(N), -np.log10(Pe_upgrade), '-ro')
    plt.legend(['degrade method', 'upgrade method'])
    plt.xlabel('bit index')
    plt.ylabel(r'$-{\log_{10}P_e}$')
    plt.title('Comparison of bit channel error rate estimated ($\mu={:d}$)'.format(mu))
    plt.savefig('comparison between degrade and upgrade.png')

    plt.show()




