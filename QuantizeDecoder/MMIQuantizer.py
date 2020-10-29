import numpy as np
from utils import channel_transition_probability_table
from PyIBQuantizer.inf_theory_tools import mutual_information

class MMIQunatizer():

    def __init__(self, prior_x1=0.5, prior_x2=0.5):
        self.px1 = prior_x1
        self.px_minus1 = prior_x2

    def channel_transition_probability_table(self, M, mu, sigma):
        '''
        determine channel transition probability table given channel parameters using discrete approximation
        :param mu:
        :param sigma:
        :return:
        '''
        delta = 0.0001
        x_continuous = np.arange(self.lowest, self.highest + delta, delta)
        x_continuous[-1] = self.lowest
        pyx_continuous = 1/np.sqrt(2*np.pi*sigma)*np.exp(-(x_continuous - mu)**2/(2*sigma))
        x_discrete = np.linspace(self.lowest, self.highest, M + 1)
        pyx = np.zeros(M)
        for i in range(M):
            index1 = x_continuous >= x_discrete[i]
            index2 = x_continuous <= x_discrete[i+1]
            index = np.bitwise_and(index1, index2)
            density = pyx_continuous[index]
            pyx[i] = np.sum(density) * delta
        return pyx, x_discrete


    def compute_partial_entropy(self, pyx_1, pyx_minus_1):
        '''
        compute partial mutual information defined in (25)
        :param pyx_1:
        :param pyx_minus_1:
        :return:
        '''
        sum_conditional_probability1 = np.sum(pyx_1)
        sum_conditional_probability2 = np.sum(pyx_minus_1)

        px1 = self.px1
        px_minus_1 = self.px_minus1

        p_nominator = px1*sum_conditional_probability1 + px_minus_1*sum_conditional_probability2

        if (p_nominator == 0):
            return 0

        p1 = sum_conditional_probability1 / p_nominator
        p2 = sum_conditional_probability2 / p_nominator

        if p1 == 0:
            tmp1 = 0
        else:
            tmp1 = pyx_1 * np.log2(p1)

        if p2 == 0:
            tmp2 = 0
        else:
            tmp2 = pyx_minus_1 * np.log2(p2)

        conditional_entropy = px1 * np.sum(tmp1) + px_minus_1 * np.sum(tmp2)
        return conditional_entropy


    def precompute_partial_entropy_table(self, M, K):
        '''
        compute partial_entropy for each quantization possibility, table[a', a] correspond to l(a', a) in paper
        l(a', a) : partial mutual information if quantize a'->a to a symbol in the alphabets of Z
        :return:
        '''
        table = np.zeros((M, M + 1))
        for a_prime in range(M):
            max_a = np.min([a_prime + M - K + 1, M])
            for a in range(a_prime + 1, max_a + 1):
                pyx_1 = self.pyx_1[a_prime:a]
                pyx_minus_1 = self.pyx_minus1[a_prime:a]
                table[a_prime, a] = self.compute_partial_entropy(pyx_1, pyx_minus_1)
        return table


    def find_opt_quantizer_AWGN(self, M, K, mu, sigma, high=2, low=-2):
        '''
        find the channel quantizer Q that maximize mutual information between transmitted symbols x and quantized symbol
        z using dynamic programming
        :return:
        '''
        self.highest = mu + 3 * sigma
        self.lowest = mu - 3 * sigma

        self.pyx_1, self.interval_x = self.channel_transition_probability_table(M, mu, sigma)
        self.pyx_minus1, _ = self.channel_transition_probability_table(M, -mu, sigma)
        self.partial_entropy_table = self.precompute_partial_entropy_table(M, K)
        Az = np.zeros(K + 1, dtype=int)
        Az[-1] = M
        state_table = np.zeros((M - K + 1, K + 1))
        local_max = -np.ones_like(state_table) * np.inf
        state_table[:, 1] = self.partial_entropy_table[0, 1:M-K+2]
        local_max[:, 1] = 0

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
                        tmp[cnt] = state_table[a_prime - a_prime_begin, z - 1] + self.partial_entropy_table[a_prime, a]
                        tmp_idx.append(a_prime)
                        cnt += 1
                    local_max[a_idx, z] = tmp_idx[np.argmax(tmp)]
                    state_table[a_idx, z] = np.max(tmp)
            else:
                a = M
                a_prime_begin = z - 1
                a_prime_end = a - 1
                tmp = np.zeros(a_prime_end - a_prime_begin + 1)
                tmp_idx = []
                cnt = 0
                for a_prime in range(a_prime_begin, a_prime_end + 1):
                    tmp[cnt] = state_table[a_prime - a_prime_begin, z - 1] + self.partial_entropy_table[a_prime, a]
                    tmp_idx.append(a_prime)
                    cnt += 1
                state_table[-1, z] = np.max(tmp)
                local_max[-1, z] = tmp_idx[np.argmax(tmp)]

        # backward tracing
        Az[-2] = int(local_max[-1, -1])
        opt_idx = int(local_max[-1, -1])
        for z in range(K - 1, 1, -1):
            opt_idx = int(local_max[opt_idx - z, z]) #
            Az[z - 1] = int(opt_idx)

        # Q matrix generation
        Q = np.zeros((K, M))
        for i in range(K):
            begin = Az[i]
            end = Az[i+1]
            Q[i, begin:end] = 1

        # P(z|x) evaluation
        pzx = np.zeros((2, K))
        for i in range(K):
            begin = Az[i]
            end = Az[i+1]
            pzx[0, i] = np.sum(self.pyx_1[begin:end])
            pzx[1, i] = np.sum(self.pyx_minus1[begin:end])

        return Az


    def find_opt_quantizer(self, joint_prob, K):

        llr = np.log2(joint_prob[0]/joint_prob[1])

        # permute the joint symbols in order to meet the llr ascending requirement for the quantizer design algorithm
        permutation = np.argsort(llr)

        # precompute partial mutual information table
        self.pyx_1 = joint_prob[0][permutation]
        self.pyx_minus1 = joint_prob[1][permutation]
        M = joint_prob.shape[1]
        self.partial_entropy_table = self.precompute_partial_entropy_table(M, K)

        # quantizer design algorithm begins
        Az = np.zeros(K + 1, dtype=int)
        Az[-1] = M
        state_table = np.zeros((M - K + 1, K + 1))
        local_max = -np.ones_like(state_table) * np.inf
        state_table[:, 1] = self.partial_entropy_table[0, 1:M - K + 2]
        local_max[:, 1] = 0

        # dynamic programming begin
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
                        tmp[cnt] = state_table[a_prime - a_prime_begin, z - 1] + self.partial_entropy_table[a_prime, a]
                        tmp_idx.append(a_prime)
                        cnt += 1
                    local_max[a_idx, z] = tmp_idx[np.argmax(tmp)]
                    state_table[a_idx, z] = np.max(tmp)
            else:
                a = M
                a_prime_begin = z - 1
                a_prime_end = a - 1
                tmp = np.zeros(a_prime_end - a_prime_begin + 1)
                tmp_idx = []
                cnt = 0
                for a_prime in range(a_prime_begin, a_prime_end + 1):
                    tmp[cnt] = state_table[a_prime - a_prime_begin, z - 1] + self.partial_entropy_table[a_prime, a]
                    tmp_idx.append(a_prime)
                    cnt += 1
                state_table[-1, z] = np.max(tmp)
                local_max[-1, z] = tmp_idx[np.argmax(tmp)]


        # backward tracing
        Az[-2] = int(local_max[-1, -1])
        opt_idx = int(local_max[-1, -1])
        for z in range(K - 1, 1, -1):
            opt_idx = int(local_max[opt_idx - z, z])  #
            Az[z - 1] = int(opt_idx)

        # Q matrix generation
        Q = np.zeros((K, M))
        for i in range(K):
            begin = Az[i]
            end = Az[i + 1]
            # for general Q matrix, permutation of input symbols should be taken into consideration
            Q[i, permutation[begin:end]] = 1
        return Q


if __name__ == "__main__":
    sigma2 = 1
    sigma = np.sqrt(sigma2)
    K = 16
    highest = 1 + 3 * sigma
    lowest = -1 - 3 * sigma
    ChannelQuantizer = MMIQunatizer(prior_x1=0.5, prior_x2=0.5)
    pyx1, interval_x = channel_transition_probability_table(128, lowest, highest, 1, sigma)
    pyx_minus1, _ = channel_transition_probability_table(128, lowest, highest, -1, sigma)
    channel_lut = ChannelQuantizer.find_opt_quantizer_AWGN(128, K, 1, sigma)
    pzx = np.zeros((2, int(K)))
    for i in range(int(K)):
        begin = channel_lut[i]
        end = channel_lut[i + 1]
        pzx[0, i] = np.sum(pyx1[begin:end])
        pzx[1, i] = np.sum(pyx_minus1[begin:end])
    I = mutual_information(0.5*pzx.transpose())
    print("I = {:f}".format(I))








