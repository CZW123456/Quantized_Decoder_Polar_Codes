import numpy as np

class SCLUTLLRDecoder():

    def __init__(self, N, K, frozen_bits, message_bits, lut_fs, lut_gs, llr_quanta):
        self.N = N
        self.K = K
        self.frozen_bits = frozen_bits
        self.msg_bits = message_bits
        self.lut_fs = lut_fs
        self.lut_gs = lut_gs
        self.llr_quanta = llr_quanta

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
                if node == self.N - 1:
                    done = True
                else:
                    node = node // 2
                    depth -= 1
            else:
                node_posi = 2 ** depth - 1 + node  # node index in the binary tree

                if node_state[node_posi] == 0:  # 0 means this node is first achieved in the traversal, calc f value
                    temp = 2 ** (n - depth)
                    incoming_symbols = symbols[depth, temp * node : temp * (node + 1)]
                    a = incoming_symbols[:temp//2]
                    b = incoming_symbols[temp//2:]

                    pairs = np.zeros((temp//2, 2))
                    pairs[:, 0] = a
                    pairs[:, 1] = b
                    pairs = pairs.astype(np.int)

                    node *= 2
                    depth += 1
                    temp //= 2

                    lut_f = np.array(self.lut_fs[node_posi])

                    if depth < n:
                        symbols[int(depth), temp * node:temp * (node + 1)] = lut_f[np.arange(temp), pairs[:, 0], pairs[:, 1]]
                    else:
                        if node in self.frozen_bits:
                            ucap[n, node] = 0
                        else:
                            symbol = int(lut_f[0, pairs[0, 0], pairs[0, 1]])
                            llr = self.llr_quanta[-1, node, symbol]
                            ucap[n, node] = int(llr < 0)

                    node_state[node_posi] = 1

                elif node_state[node_posi] == 1:
                    temp = 2 ** (n - depth)
                    incoming_symbols = symbols[depth, temp * node: temp * (node + 1)]
                    a = incoming_symbols[:temp//2]
                    b = incoming_symbols[temp//2:]

                    pairs = np.zeros((temp//2, 2))
                    pairs[:, 0] = a
                    pairs[:, 1] = b
                    pairs = pairs.astype(np.int)

                    ltemp = temp // 2
                    lnode = 2 * node
                    cdepth = depth + 1
                    ucapl = ucap[cdepth, ltemp * lnode : ltemp * (lnode + 1)].astype(np.int)  # incoming decision from the left child

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
                            llr = self.llr_quanta[-1, node, symbol]
                            ucap[n, node] = int(llr < 0)

                    node_state[node_posi] = 2
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
        msg_bits = ucap[n, self.msg_bits]
        msg_bits = msg_bits.astype(np.int)
        return msg_bits