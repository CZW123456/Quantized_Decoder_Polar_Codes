import numpy as np
class SCLDecoder():

    def __init__(self, N, K, L, frozenbit, msgbits):
        self.N = N  # code length
        self.K = K  # msg bits length + crc length
        self.L = L  # list size
        self.frozen_bits = frozenbit  # position of frozen bits
        self.msg_posi = msgbits  # position of msg bits

    def f(self, a, b):
        L = a.shape[0]
        abs_min = np.min(np.concatenate([np.abs(a), np.abs(b)], axis=1).reshape((L, 2, a.shape[1])), axis=1)
        return (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * abs_min

    def g(self, a, b, c):
        return b + (1 - 2 * c) * a

    def mink(self, a, k):
        idx = np.argsort(a)[:k]
        return a[idx], idx

    def decode(self, channel_vector_llr):
        n = int(np.log2(self.N))  # depth of the binary tree
        LLR = np.zeros((self.L, n + 1, self.N))  # LLR matrix for decoding by L decoders
        LLR[:, 0, :] = channel_vector_llr  # initialize the llr in L root nodes
        ucap = np.zeros((self.L, n + 1, self.N))  # upward decision result for reverse binary tree traversal
        node_state = np.zeros(2 * self.N - 1)  # node probab
        PML = np.ones(self.L) * np.inf  # Path Metric
        PML[0] = 0
        depth = 0
        node = 0
        done = False

        # begin Polar SCL decoding by binary tree traversal and path expansion and pruning
        while done == False:
            # check leaf or not
            # leaf node
            if depth == n:

                DM = LLR[:, n, node]
                # check wether this is a frozen bit
                # for frozen bit, it must be zero
                if node in self.frozen_bits:
                    ucap[:, n, node] = 0
                    PML += np.abs(DM) * (DM < 0)  # if DM is negative, add |DM|
                # for information bit, it should be hard decided by incoming LLR
                else:
                    decision = DM < 0  # if LLR < 0 then it should be 1, else be 0
                    PM2 = np.concatenate([PML, PML + abs(DM)])  # merge historical PM and current PM + DM
                    PML, posi = self.mink(PM2, self.L)  # each non frozen leaf node a ascending sort happens
                    posi1 = posi >= self.L
                    posi[posi1] -= self.L  # adjust position so that they can remain smaller than L
                    decision = decision[posi]
                    decision[posi1] = 1 - decision[posi1]
                    LLR = LLR[posi, :, :]  # rearrange LLR tensor
                    ucap = ucap[posi, :, :]
                    ucap[:, n, node] = decision

                if node == self.N - 1:  # if this is the last bit to be decoded
                    done = True
                else:  # if this is not the last bit to be decoded, it should return to his parent node
                    node = node // 2
                    depth -= 1

            else:  # other intermediate node
                node_posi = 2 ** depth - 1 + node  # node index in the binary tree
                if node_state[node_posi] == 0:  # 0 means this node is first achieved in the traversal, calc f value
                    temp = 2 ** (n - depth)
                    incoming_llr = LLR[:, depth, temp * node: temp * (node + 1)]
                    a = incoming_llr[:, :temp // 2]
                    b = incoming_llr[:, temp // 2:]
                    # calc location for the left child
                    node *= 2
                    depth += 1
                    # length of the incoming belief vector of the left node
                    temp /= 2
                    LLR[:, depth, int(temp) * node: int(temp) * (node + 1)] = self.f(a, b)  # update the LLR in current node
                    node_state[node_posi] = 1  # switch node state from 0 to 1

                elif node_state[node_posi] == 1:  # 1 means this node is visited once, and it should visit its right child, calc g value

                    temp = 2 ** (n - depth)
                    incoming_llr = LLR[:, depth, int(temp) * node: int(temp) * (node + 1)]
                    a = incoming_llr[:, :temp // 2].squeeze()
                    b = incoming_llr[:, temp // 2:].squeeze()
                    ltemp = temp // 2
                    lnode = 2 * node
                    cdepth = depth + 1
                    ucapl = ucap[:, cdepth, ltemp * lnode: ltemp * (lnode + 1)].squeeze()  # incoming decision from the left child
                    node = 2 * node + 1
                    depth += 1
                    temp /= 2
                    g_res = self.g(a, b, ucapl)
                    if len(g_res.shape) == 1:
                        g_res = g_res.reshape((g_res.shape[0], 1))
                    LLR[:, depth, int(temp) * node: int(temp) * (node + 1)] = g_res
                    node_state[node_posi] = 2  # switch node state from 1 to 2

                else:  # left and right child both have been traversed, now summarize decision from the two nodes to the parent
                    temp = 2 ** (n - depth)
                    ctemp = temp // 2
                    lnode = 2 * node
                    rnode = 2 * node + 1
                    cdepth = depth + 1
                    ucapl = ucap[:, cdepth, ctemp * lnode: ctemp * (lnode + 1)]
                    ucapr = ucap[:, cdepth, ctemp * rnode: ctemp * (rnode + 1)]
                    ucap[:, depth, int(temp) * node: int(temp) * (node + 1)] = np.concatenate([np.mod(ucapl + ucapr, 2), ucapr], axis=1)  # summarize function
                    node = node // 2
                    depth -= 1

        idx = np.argmin(PML)
        decoded_bits = ucap[idx, n, self.msg_posi]
        return decoded_bits










