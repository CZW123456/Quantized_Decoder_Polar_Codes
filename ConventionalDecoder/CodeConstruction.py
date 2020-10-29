import numpy as np

def derivative_phi(x):
    '''
    derivative of Pi
    :param x:
    :return:
    '''
    if x >= 0 and x <= 10:
        dx = -0.4527*0.86*np.power(x, -0.14) * phi(x)
    else:
        dx = np.exp(-x/4)*np.sqrt(np.pi/x)*(-1/2/x*(1 - 10/7/x) - 1/4*(1 - 10/7/x) + 10/7/x/x)
    return dx

def phi_inverse(x):
    '''
    calc phi inverse using fix point iteration
    :param x:
    :return:
    '''
    if x >= 0.0388 and x <= 1.0221:
        return np.power((0.0218 - np.log(x)) / 0.4527, 1 / 0.86)
    else:
        x0 = 0.0388
        x1 = x0 - (phi(x0) - x) / derivative_phi(x0)
        delta = abs(x1 - x0)
        epsilon = 1e-3
        num_iter = 0
        while (delta >= epsilon):
            num_iter = num_iter + 1
            x0 = x1
            x1 = x1 - (phi(x1) - x) / derivative_phi(x1)
            if x1 > 1e2:
                epsilon = 10
            delta = abs(x1 - x0)
        return x1

def phi(x):
    '''
    evaluate phi function
    :param x:
    :return:
    '''
    if x >= 0 and x <= 10:
        return np.exp(-0.4527 * np.power(x, 0.859) + 0.0218)
    elif x > 10:
        return np.sqrt(np.pi/x) * np.exp(-x/4) * (1 - 10/7/x)
    else:
        NotImplementedError

def bitreverse(x):
    if len(x) == 1:
        return x
    else:
        l = len(x)
        a = x[0:l:2]
        b = x[1:l:2]
        a = bitreverse(a)
        b = bitreverse(b)
        y = np.concatenate([a, b])
        return y

class PolarCodeConstructor():

    def __init__(self, N, K, QPath):
        self.N = N
        self.K = K
        reliable_sequence = np.loadtxt(QPath, delimiter="\n").astype(np.int)
        self.Q1 = reliable_sequence[reliable_sequence < N]

    def PW(self):
        '''
        Code construction with 5G NR standard
        :return:
        '''
        frozenbits = np.sort(self.Q1[:self.N - self.K])
        msgbits = np.array([i for i in range(self.N) if i not in frozenbits])

        frozen_mask = np.zeros(self.N).astype(np.int)
        message_mask = np.zeros(self.N).astype(np.int)
        frozen_mask[frozenbits] = 1
        message_mask[msgbits] = 1

        return frozenbits, msgbits, frozen_mask, message_mask

    def GA(self, sigma):
        n = int(np.log2(self.N))
        u = np.zeros((n + 1, self.N))
        u[0, :] = 2 / sigma**2
        for level in range(1, n + 1):
            num_bits_parent_node = 2 ** (n - level + 1)
            num_bits_cur_node = num_bits_parent_node // 2
            num_node_parent_level = 2 ** (level - 1)
            for node in range(num_node_parent_level):
                lnode = 2 * node
                rnode = 2 * node + 1
                poffset = num_bits_parent_node * node
                tmp = u[level - 1, poffset]
                f = phi_inverse(1 - (1 - phi(tmp))**2)
                g = 2 * tmp
                u[level, lnode*num_bits_cur_node:(lnode+1)*num_bits_cur_node] = f
                u[level, rnode*num_bits_cur_node:(rnode+1)*num_bits_cur_node] = g

        # select the most reliable bit channel
        channel_quality_idx_up = np.argsort(bitreverse(u[-1, :]))
        frozenbits = channel_quality_idx_up[:self.N - self.K]
        msgbits = channel_quality_idx_up[self.N - self.K:]
        frozen_mask = np.zeros(self.N).astype(np.int)
        message_mask = np.zeros(self.N).astype(np.int)
        frozen_mask[frozenbits] = 1
        message_mask[msgbits] = 1

        # save E(LLR)
        self.E_LLR = u
        return frozenbits, msgbits, frozen_mask, message_mask









