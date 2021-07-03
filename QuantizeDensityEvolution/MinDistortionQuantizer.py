import numpy as np

def compute_partial_quantization_noise(density, quanta):
    assert density.shape == quanta.shape
    new_quanta = np.sum(density * quanta) / np.sum(density)
    quantization_noise = np.sum((quanta - new_quanta)**2 * density)
    return quantization_noise


def precompute_quantization_noise_table(density, quanta, M, K):
    '''
    Compute the quantization noise of each possible quantization scheme, table[a', a] corresponding to the quantization
    noise if quantize a'->a to a symbol in the alphabet of Z
    :param density:
    :param quanta:
    :param M:
    :param K:
    :return:
    '''
    table = np.zeros((M, M + 1))
    for a_prime in range(M):
        max_a = np.min([a_prime + M - K + 1, M])
        for a in range(a_prime + 1, max_a + 1):
            table[a_prime, a] = compute_partial_quantization_noise(density[a_prime:a], quanta[a_prime:a])
    return table


def find_OptLS_quantizer(llr_density, llr_quanta, K):

    M = llr_density.shape[0]

    permutation = np.argsort(llr_quanta)

    llr_density = llr_density[permutation]
    llr_quanta = llr_quanta[permutation]

    # Pre-compute the cost of quantizing a'->a to a symbol z
    quantization_noise_table = precompute_quantization_noise_table(density=llr_density, quanta=llr_quanta, M=M, K=K)

    # Quantizer design begins
    Az = np.zeros(K + 1).astype(np.int32)
    Az[-1] = M

    state_table = np.zeros((M - K + 1, K + 1))
    local_min = np.ones_like(state_table) * np.inf
    state_table[:, 1] = quantization_noise_table[0, 1:M - K + 2]
    local_min[:, 1] = 0

    # Dynamic programming begins

    # forward computing
    for z in range(2, K + 1):
        if z < K:
            for a in range(z, z + M - K + 1):
                a_idx = a - z
                a_prime_begin = z - 1
                a_prime_end = a - 1
                tmp = np.zeros(a_prime_end - a_prime_begin + 1)
                tmp_idx = []
                cnt = 0
                for a_prime in range(a_prime_begin, a_prime_end + 1):
                    tmp[cnt] = state_table[a_prime - a_prime_begin, z - 1] + quantization_noise_table[a_prime, a]
                    tmp_idx.append(a_prime)
                    cnt += 1
                local_min[a_idx, z] = tmp_idx[np.argmin(tmp)]
                state_table[a_idx, z] = np.min(tmp)
        else:
            a = M
            a_prime_begin = z - 1
            a_prime_end = a - 1
            tmp = np.zeros(a_prime_end - a_prime_begin + 1)
            tmp_idx = []
            cnt = 0
            for a_prime in range(a_prime_begin, a_prime_end + 1):
                tmp[cnt] = state_table[a_prime - a_prime_begin, z - 1] + quantization_noise_table[a_prime, a]
                tmp_idx.append(a_prime)
                cnt += 1
            state_table[-1, z] = np.min(tmp)
            local_min[-1, z] = tmp_idx[np.argmin(tmp)]

    # backward tracing
    Az[-2] = int(local_min[-1, -1])
    opt_idx = int(local_min[-1, -1])
    for z in range(K - 1, 1, -1):
        opt_idx = int(local_min[opt_idx - z, z])  #
        Az[z - 1] = int(opt_idx)

    # LUT generation
    lut = np.zeros(M).astype(np.int32)
    quanta = np.zeros(K).astype(np.float64)
    density = np.zeros(K).astype(np.float64)
    for i in range(K):
        begin = Az[i]
        end = Az[i + 1]
        lut[permutation[begin:end]] = i
        quanta[i] = np.sum(llr_quanta[permutation[begin:end]] * llr_density[permutation[begin:end]]) / np.sum(llr_density[permutation[begin:end]])
        density[i] = np.sum(llr_density[permutation[begin:end]])

    return density, quanta, lut



