import numpy as np

class SCDecoder():

    def __init__(self, N, K, frozen_bits, msg_bits, channel_level=None, channel_idx=None):
        self.N = N
        self.K = K
        self.frozen_bits = frozen_bits
        self.msg_posi = msg_bits
        self.channel_level = channel_level
        self.channel_idx = channel_idx

    def f(self, a, b):
        return (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * np.min(np.concatenate([np.abs(a), np.abs(b)]).reshape((2, a.shape[0])), axis=0)

    def g(self, a, b, c):
        return b + (1 - 2 * c) * a


    def decode(self, channel_vector_llr):
        n = int(np.log2(self.N))
        LLR = np.zeros((n + 1, self.N))  # LLR matrix for decoding
        LLR[0] = channel_vector_llr  # initialize the llr in the root node
        ucap = np.zeros((n + 1, self.N)) # upward decision result for reverse binary tree traversal
        node_state = np.zeros(2 * self.N - 1)  # node probab
        depth = 0
        node = 0
        done = False
        # begin Polar SC decoding by binary tree traversal
        while done == False:
            # check leaf or not
            # leaf node
            if depth == n:
                # check wether this is a frozen bit
                # for frozen bit, it must be zero
                if node in self.frozen_bits:
                    ucap[n, node] = int(0)
                # for information bit, it should be hard decided by incoming LLR
                else:
                    if LLR[n, node] >= 0:
                        ucap[n, node] = int(0)
                    else:
                        ucap[n, node] = int(1)
                if node == self.N - 1: # if this is the last bit to be decoded
                    done = True
                else: # if this is not the last bit to be decoded, it should return to his parent node
                    node = node // 2
                    depth -= 1
            else: # other intermediate node
                node_posi = 2 ** depth - 1 + node  # node index in the binary tree
                if node_state[node_posi] == 0:  # 0 means this node is first achieved in the traversal, calc f value
                    temp = 2 ** (n - depth)
                    incoming_llr = LLR[depth, temp * node : temp * (node + 1)]
                    a = incoming_llr[:temp//2]
                    b = incoming_llr[temp//2:]
                    # calc location for the left child
                    node *= 2
                    depth += 1
                    # length of the incoming belief vector of the left node
                    temp /= 2
                    LLR[int(depth), int(temp * node) : int(temp) * int(node + 1)] = self.f(a, b)  # update the LLR in current node
                    node_state[node_posi] = 1  # switch node state from 0 to 1
                elif node_state[node_posi] == 1: # 1 means this node is visited once, and it should visit its right child, calc g value
                    temp = 2 ** (n - depth)
                    incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                    a = incoming_llr[:temp // 2]
                    b = incoming_llr[temp // 2:]
                    ltemp = temp // 2
                    lnode = 2 * node
                    cdepth = depth + 1
                    ucapl = ucap[cdepth, ltemp * lnode : ltemp * (lnode + 1)]  # incoming decision from the left child
                    node = 2 * node + 1
                    depth += 1
                    temp /= 2
                    LLR[depth, int(temp) * node : int(temp) * (node + 1)] = self.g(a, b, ucapl)
                    node_state[node_posi] = 2 # switch node state from 1 to 2
                else:  # left and right child both have been traversed, now summarize decision from the two nodes to the parent
                    temp = 2 ** (n - depth)
                    ctemp = temp // 2
                    lnode = 2 * node
                    rnode = 2 * node + 1
                    cdepth = depth + 1
                    ucapl = ucap[cdepth, ctemp * lnode : ctemp * (lnode + 1)]
                    ucapr = ucap[cdepth, ctemp * rnode : ctemp * (rnode + 1)]
                    ucap[depth, int(temp) * node : int(temp) * (node + 1)] = np.concatenate([np.mod(ucapl + ucapr, 2), ucapr], axis=0)  # summarize function
                    node = node // 2
                    depth -= 1

        # SC decoding end
        msg_bits = ucap[n, self.msg_posi]
        return msg_bits, LLR[self.channel_level, self.channel_idx]














