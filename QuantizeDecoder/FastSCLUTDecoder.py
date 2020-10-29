import numpy as np

class FastSCLUTDecoder():

    def __init__(self, N, K, frozen_bits, message_bits, node_type, lut_fs, lut_gs, virtual_channel_llr):
        self.N = N
        self.K = K
        self.frozen_bits = frozen_bits
        self.msg_bits = message_bits
        self.node_type = node_type
        self.lut_fs = lut_fs
        self.lut_gs = lut_gs
        self.virtual_channel_llr = virtual_channel_llr

    def decode(self, channel_quantized_symbols):
        n = int(np.log2(self.N))
        symbols = np.zeros((n + 1, self.N))     # LLR matrix for decoding
        symbols[0] = channel_quantized_symbols  # initialize the llr in the root node
        ucap = np.zeros((n + 1, self.N))        # upward decision result for reverse binary tree traversal
        node_state = np.zeros(self.N - 1)       # node state
        depth = 0
        node = 0
        done = False

        while done == False:
            if depth == n:
                node = node // 2
                depth -= 1
            else:
                node_posi = int(2 ** depth - 1 + node)  # node index in the binary tree

                if node_state[node_posi] == 0:

                    # R0 node
                    if self.node_type[node_posi] == 0:
                        temp = 2 ** (n - depth)
                        ucap[depth, temp * node: temp * (node + 1)] = np.zeros(temp)
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    # R1 node
                    if self.node_type[node_posi] == 1:
                        temp = 2 ** (n - depth)
                        incoming_symbols = symbols[depth, temp * node: temp * (node + 1)].astype(np.int)
                        llr = self.virtual_channel_llr[depth - 1, np.arange(temp*node, temp*(node+1)), incoming_symbols]
                        decision = (1 - np.sign(llr)) / 2
                        ucap[depth, temp * node: temp * (node + 1)] = decision
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    # REP node
                    if self.node_type[node_posi] == 2:
                        temp = 2 ** (n - depth)
                        incoming_symbols = symbols[depth, temp * node: temp * (node + 1)].astype(np.int)
                        llr = self.virtual_channel_llr[depth - 1, np.arange(temp * node, temp * (node + 1)), incoming_symbols]
                        S = np.sum(llr)
                        if S > 0:
                            decision = np.zeros(temp)
                        else:
                            decision = np.ones(temp)
                        ucap[depth, temp * node: temp * (node + 1)] = decision
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    # SPC node
                    if self.node_type[node_posi] == 3:
                        temp = 2 ** (n - depth)
                        incoming_symbols = symbols[depth, temp * node: temp * (node + 1)].astype(np.int)
                        llr = self.virtual_channel_llr[depth - 1, np.arange(temp * node, temp * (node + 1)), incoming_symbols]
                        decision = (1 - np.sign(llr)) / 2
                        decision = decision.astype(np.bool)
                        parity_check = np.mod(np.sum(decision), 2)
                        if parity_check == 0:
                            ucap[depth, temp * node: temp * (node + 1)] = decision
                        else:
                            min_abs_llr_idx = np.argmin(np.abs(llr))
                            decision[min_abs_llr_idx] = np.mod(decision[min_abs_llr_idx] + 1, 2)  # filp the bit with minimun absolute LLR
                            ucap[depth, temp * node: temp * (node + 1)] = decision
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    temp = 2 ** (n - depth)
                    incoming_symbols = symbols[depth, temp * node: temp * (node + 1)]
                    a = incoming_symbols[:temp // 2]
                    b = incoming_symbols[temp // 2:]

                    pairs = np.zeros((temp // 2, 2))
                    pairs[:, 0] = a
                    pairs[:, 1] = b
                    pairs = pairs.astype(np.int)

                    node *= 2
                    depth += 1
                    temp //= 2

                    lut_f = np.array(self.lut_fs[node_posi])

                    if depth < n:
                        symbols[int(depth), temp * node:temp * (node + 1)] = lut_f[
                            np.arange(temp), pairs[:, 0], pairs[:, 1]]
                    else:
                        if node in self.frozen_bits:
                            ucap[n, node] = 0
                        else:
                            symbol = int(lut_f[0][pairs[0, 0], pairs[0, 1]])
                            llr = self.virtual_channel_llr[-1, node, symbol]
                            ucap[n, node] = int(llr < 0)

                    node_state[node_posi] = 1

                elif node_state[node_posi] == 1:
                    temp = 2 ** (n - depth)
                    incoming_symbols = symbols[depth, temp * node: temp * (node + 1)]
                    a = incoming_symbols[:temp // 2]
                    b = incoming_symbols[temp // 2:]

                    pairs = np.zeros((temp // 2, 2))
                    pairs[:, 0] = a
                    pairs[:, 1] = b
                    pairs = pairs.astype(np.int)

                    ltemp = temp // 2
                    lnode = 2 * node
                    cdepth = depth + 1
                    ucapl = ucap[cdepth, ltemp * lnode: ltemp * (lnode + 1)].astype(np.int)  # incoming decision from the left child

                    node = 2 * node + 1
                    depth += 1
                    temp //= 2

                    lut_g = np.array(self.lut_gs[node_posi])

                    if depth < n:
                        symbols[int(depth), temp * node:temp * (node + 1)] = lut_g[np.arange(temp), ucapl, pairs[:, 0], pairs[:, 1]]
                    else:
                        if node in self.frozen_bits:
                            ucap[n, node] = 0
                        else:
                            symbol = int(lut_g[0][int(ucapl[0]), pairs[0, 0], pairs[0, 1]])
                            llr = self.virtual_channel_llr[-1, node, symbol]
                            ucap[n, node] = int(llr < 0)
                    node_state[node_posi] = 2

                else:  # left and right child both have been traversed, now summarize decision from the two nodes to the parent
                    temp = 2 ** (n - depth)
                    ctemp = temp // 2
                    lnode = 2 * node
                    rnode = 2 * node + 1
                    cdepth = depth + 1
                    ucapl = ucap[cdepth, ctemp * lnode: ctemp * (lnode + 1)]
                    ucapr = ucap[cdepth, ctemp * rnode: ctemp * (rnode + 1)]
                    ucap[depth, int(temp) * node: int(temp) * (node + 1)] = np.concatenate(
                        [np.mod(ucapl + ucapr, 2), ucapr], axis=0)  # summarize function
                    if node == 0 and depth == 0:  # if this is the last bit to be decoded
                        done = True
                    else:
                        node = node // 2
                        depth -= 1

        # FastSCLUT decoding end
        x = ucap[0]  # obtain decoding result in the channel part, not the transmitter part
        m = 1
        # recode the decoded channel part code word to obtain the ultimate transmitted bits
        for d in range(n - 1, -1, -1):
            for i in range(0, self.N, 2 * m):
                a = x[i: i + m]  # first part
                b = x[i + m: i + 2 * m]  # second part
                x[i: i + 2 * m] = np.concatenate([np.mod(a + b, 2), b])  # combining
            m *= 2
        u = x[self.msg_bits]
        return u