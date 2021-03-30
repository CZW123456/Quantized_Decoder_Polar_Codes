import numpy as np
import copy
from ConventionalDecoder.IdentifyNodes import NodeIdentifier

class FastCASCLDecoder():

    def __init__(self, N, K, A, L, frozenbit, msgbits, node_type=None):
        self.N = N                      # code length
        self.K = K                      # msg bits length + crc length
        self.A = A                      # msg bits length
        self.L = L                      # list size
        self.frozen_bits = frozenbit    # position of frozen bits
        self.msg_posi = msgbits         # position of msg bits
        self.node_type = node_type      # node type: R0:0, R1:1, REP:2, SPC:3

    def f(self, a, b):
        L = a.shape[0]
        abs_min = np.min(np.concatenate([np.abs(a), np.abs(b)], axis=1).reshape((L, 2, a.shape[1])), axis=1)
        return (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * abs_min

    def g(self, a, b, c):
        return b + (1 - 2 * c) * a

    def mink(self, a, k):
        idx = np.argsort(a)[:k]
        return a[idx], idx

    def decode(self, channel_vector_llr, crc=None):
        n = int(np.log2(self.N))  # depth of the binary tree
        LLR = np.zeros((self.L, n + 1, self.N))  # LLR matrix for decoding by L decoders
        LLR[:, 0, :] = channel_vector_llr  # initialize the llr in L root nodes
        ucap = np.zeros((self.L, n + 1, self.N))  # upward decision result for reverse binary tree traversal
        node_state = np.zeros(2 * self.N - 1)  # node probab
        PML = np.ones(self.L) * 1e300  # Path Metric
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
                if node in self.frozen_bits:
                    ucap[:, n, node] = 0
                    PML += np.abs(DM) * (DM < 0)  # if DM is negative, add |DM|
                # for information bit, it should be hard decided by incoming LLR
                else:
                    decision = DM < 0  # hard decision
                    PM2 = np.concatenate([PML, PML + abs(DM)])  # merge historical PM and current PM + DM
                    PML, posi = self.mink(PM2, self.L)  # each non frozen leaf node a ascending sort happens
                    posi1 = posi >= self.L
                    posi[posi1] -= self.L  # adjust position so that they can remain smaller than L
                    decision = decision[posi]
                    decision[posi1] = 1 - decision[posi1]
                    LLR = LLR[posi, :, :]  # rearrange LLR tensor
                    ucap = ucap[posi, :, :]
                    ucap[:, n, node] = decision
                # return to parent node
                node = node // 2
                depth -= 1
            else:  # other intermediate node
                node_posi = 2 ** depth - 1 + node  # node index in the binary tree

                if node_state[node_posi] == 0:  # 0 means this node is first achieved in the traversal, calc f value
                    temp = 2 ** (n - depth)
                    incoming_llr = LLR[:, depth, temp * node: temp * (node + 1)]

                    # check whether current node belongs to the predefined special node
                    if self.node_type[node_posi] == 0:
                        # R0 node
                        ucap[:, depth, temp * node : temp * (node + 1)] = 0
                        PML += np.sum((incoming_llr < 0) * np.abs(incoming_llr), axis=1)
                        # return to parent node
                        node = node // 2
                        depth -= 1
                        continue

                    if self.node_type[node_posi] == 1:
                        # R1 node
                        decision = incoming_llr < 0                   # hard decision
                        max_depth = np.min([self.L - 1, temp])        # maximum depth of the search tree
                        abs_llr = np.abs(incoming_llr)
                        sorted_llr_idx = np.argsort(abs_llr, axis=1)  # sort llr array of each SC decoder in ascending order
                        # tree search begin
                        for layer in range(max_depth):
                            l_PML = copy.deepcopy(PML)
                            r_PML = copy.deepcopy(PML)
                            for j in range(self.L):
                                r_PML[j] += abs_llr[j, sorted_llr_idx[j, layer]]
                            PML, posi = self.mink(np.concatenate([l_PML, r_PML]), self.L)
                            posi1 = posi >= self.L
                            posi[posi1] -= self.L
                            decision = decision[posi, :]
                            decision[posi1, sorted_llr_idx[posi[posi1], layer]] = 1 - decision[posi1, sorted_llr_idx[posi[posi1], layer]]
                            # rearrange
                            LLR = LLR[posi, :, :]
                            ucap = ucap[posi, :, :]
                            abs_llr = abs_llr[posi, :]
                            sorted_llr_idx = sorted_llr_idx[posi, :]

                        ucap[:, depth, temp * node : temp * (node + 1)] = decision
                        # return to parent node
                        node = node // 2
                        depth -= 1
                        continue

                    if self.node_type[node_posi] == 2:
                        # REP node
                        abs_incoming_llr = np.abs(incoming_llr)
                        penalty_0 = np.sum((incoming_llr < 0) * abs_incoming_llr, axis=1, keepdims=True)
                        penalty_1 = np.sum((incoming_llr >= 0) * abs_incoming_llr, axis=1, keepdims=True)
                        tmp = np.concatenate([PML + penalty_0.squeeze(), PML + penalty_1.squeeze()])
                        PML, posi = self.mink(tmp, self.L)
                        posi1 = posi >= self.L
                        posi[posi1] -= self.L
                        decision = np.zeros((self.L, temp))
                        decision[posi1] = 1
                        # rearrange LLR and ucap tensor——decoder selection
                        LLR = LLR[posi, :, :]
                        ucap = ucap[posi, :, :]
                        ucap[:, depth, temp * node : temp * (node + 1)] = decision
                        # return to parent node
                        depth -= 1
                        node = node // 2
                        continue

                    if self.node_type[node_posi] == 3:
                        # SPC node
                        decision = incoming_llr < 0
                        even_parity = np.mod(np.sum(decision, axis=1), 2) # even parity check
                        flip_smallest_time = np.zeros(self.L)
                        flip_smallest_time[even_parity==1] += 1
                        abs_llr = np.abs(incoming_llr)
                        sorted_llr_idx = np.argsort(abs_llr, axis=1)      # sort llr array of each SC decoder in ascending order
                        PML[even_parity == 1] += abs_llr[even_parity==1, sorted_llr_idx[even_parity==1, 0]]
                        decision[even_parity==1, sorted_llr_idx[even_parity==1, 0]] = 1 - decision[even_parity==1, sorted_llr_idx[even_parity==1, 0]]
                        max_depth = np.min([self.L, temp])

                        # tree search
                        for layer in range(1, max_depth):
                            l_PML = copy.deepcopy(PML)
                            r_PML = copy.deepcopy(PML)
                            for j in range(self.L):
                                if flip_smallest_time[j] % 2 == 0:
                                    r_PML[j] += (abs_llr[j, sorted_llr_idx[j, layer]] + abs_llr[j, sorted_llr_idx[j, 0]])
                                else:
                                    r_PML[j] += (abs_llr[j, sorted_llr_idx[j, layer]] - abs_llr[j, sorted_llr_idx[j, 0]])
                            PML, posi = self.mink(np.concatenate([l_PML, r_PML]), self.L)
                            posi1 = posi >= self.L
                            posi[posi1] -= self.L
                            # flip two bits at a time
                            decision = decision[posi, :]
                            decision[posi1, sorted_llr_idx[posi[posi1], 0]] = 1 - decision[posi1, sorted_llr_idx[posi[posi1], 0]]
                            decision[posi1, sorted_llr_idx[posi[posi1], layer]] = 1 - decision[posi1, sorted_llr_idx[posi[posi1], layer]]
                            flip_smallest_time[posi1] += 1
                            # rearrange
                            LLR = LLR[posi, :, :]
                            ucap = ucap[posi, :, :]
                            abs_llr = abs_llr[posi, :]
                            sorted_llr_idx = sorted_llr_idx[posi, :]
                            even_parity = even_parity[posi]
                            flip_smallest_time = flip_smallest_time[posi]

                        ucap[:, depth, temp * node: temp * (node + 1)] = decision
                        # return to parent node
                        node = node // 2
                        depth -= 1
                        continue

                    a = incoming_llr[:, :temp// 2]
                    b = incoming_llr[:, temp//2:]
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
                    if depth == 0 and node == 0:
                        done = True
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

        # if none of the decoded bits satisfy thr CRC check then simply choose the decoded result with smllest path metric
        idx = np.argmin(PML)
        x = ucap[idx][0]  # obtain decoding result in the channel part, not the transmitter part
        m = 1
        # recode the decoded channel part code word to obtain the ultimate transmitted bits
        for d in range(n - 1, -1, -1):
            for i in range(0, self.N, 2 * m):
                a = x[i: i + m]  # first part
                b = x[i + m: i + 2 * m]  # second part
                x[i: i + 2 * m] = np.concatenate([np.mod(a + b, 2), b])  # combining
            m *= 2
        u = x[self.msg_posi]
        return u

if __name__ == "__main__":
    N = 8
    K = 4
    L = 4
    llr = np.array([-2.0,-2.5,-4.0,1.0,-6.5,6.0,16.6,3.5])
    frozen_bits = [0, 1, 2, 3, 4]
    msg_bits = [5, 6, 7]
    node_identifier = NodeIdentifier(N, K, frozen_bits, msg_bits, use_new_node=False)
    node_type = node_identifier.run()
    # node_identifier.show_traverse_path(node_type)
    SCL = FastCASCLDecoder(N, K, K, L, frozen_bits, msg_bits, node_type)
    u = SCL.decode(llr)
    print(u)










