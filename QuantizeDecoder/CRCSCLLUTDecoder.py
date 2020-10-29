import numpy as np
class CRCSCLLUTDecoder():

    def __init__(self, N, K, A, L, frozenbit, msgbits, lut_fs, lut_gs, virtual_channel_llr):
        self.N = N  # code length
        self.K = K  # msg bits length + crc length
        self.A = A  # msg bits length
        self.L = L  # list size
        self.frozen_bits = frozenbit  # position of frozen bits
        self.msg_posi = msgbits  # position of msg bits
        self.lut_fs = lut_fs
        self.lut_gs = lut_gs
        self.virtual_channel_llr = virtual_channel_llr

    def mink(self, a, k):
        idx = np.argsort(a)[:k]
        return a[idx], idx


    def decode(self, channel_quantized_symbols, crc):
        n = int(np.log2(self.N))
        symbols = np.zeros((self.L, n + 1, self.N))
        symbols[:, 0, :] = channel_quantized_symbols
        ucap = np.zeros((self.L, n + 1, self.N))
        node_state = np.zeros(2 * self.N - 1)
        PML = np.ones(self.L) * np.inf
        PML[0] = 0
        depth = 0
        node = 0
        done = False

        # begin Polar SCL decoding by binary tree traversal and path expansion and pruning
        while done == False:

            # leaf node
            if depth == n:

                if node == self.N - 1:
                    done = True
                else:
                    node = node // 2
                    depth -= 1

            else:
                node_posi = 2 ** depth - 1 + node

                if node_state[node_posi] == 0:
                    temp = int(2 ** (n - depth))
                    incoming_symbols = symbols[:, depth, temp * node: temp * (node + 1)]
                    a = incoming_symbols[:, :temp // 2]
                    b = incoming_symbols[:, temp // 2:]

                    pairs = np.zeros((self.L, temp // 2, 2))
                    pairs[:, :, 0] = a
                    pairs[:, :, 1] = b
                    pairs = pairs.astype(np.int)

                    lut_f = np.array(self.lut_fs[node_posi])

                    # compute location for the left child
                    node *= 2
                    depth += 1
                    temp /= 2

                    if depth < n:
                        for l in range(self.L):
                            symbols[l, int(depth), int(temp * node) : int(temp * (node + 1))] = lut_f[np.arange(temp).astype(np.int), pairs[l, :, 0], pairs[l, :, 1]]
                    else:
                        if node in self.frozen_bits:
                            ucap[:, n, node] = 0
                            llr = np.zeros(self.L)
                            for l in range(self.L):
                                symbol = int(lut_f[0, pairs[l, 0, 0], pairs[l, 0, 1]])
                                llr[l] = self.virtual_channel_llr[-1, node, symbol]
                            PML += np.abs(llr) * (llr < 0)  # if DM is negative, add |DM|
                        else:
                            decision = np.zeros(self.L)
                            PM2 = np.concatenate([PML, np.zeros(self.L)])
                            for l in range(self.L):
                                symbol = int(lut_f[0, pairs[l, 0, 0], pairs[l, 0, 1]])
                                llr = self.virtual_channel_llr[-1, node, symbol]
                                decision[l] = llr < 0
                                PM2[self.L + l] = PML[l] + np.abs(llr)
                            PML, posi = self.mink(PM2, self.L)
                            posi1 = posi >= self.L
                            posi[posi1] -= self.L
                            decision = decision[posi]
                            decision[posi1] = 1 - decision[posi1]
                            symbols = symbols[posi, :, :]
                            ucap = ucap[posi, :, :]
                            ucap[:, n, node] = decision

                    node_state[node_posi] = 1

                elif node_state[node_posi] == 1:

                    temp = 2 ** (n - depth)
                    incoming_symbols = symbols[:, depth, temp * node: temp * (node + 1)]
                    a = incoming_symbols[:, :temp//2]
                    b = incoming_symbols[:, temp//2:]

                    pairs = np.zeros((self.L, temp // 2, 2))
                    pairs[:, :, 0] = a
                    pairs[:, :, 1] = b
                    pairs = pairs.astype(np.int)

                    ltemp = temp // 2
                    lnode = 2 * node
                    cdepth = depth + 1
                    ucapl = ucap[:, cdepth, ltemp * lnode: ltemp * (lnode + 1)].squeeze().astype(np.int)
                    node = 2 * node + 1
                    depth += 1
                    temp /= 2

                    lut_g = np.array(self.lut_gs[node_posi])

                    if depth < n:
                        for l in range(self.L):
                            symbols[l, int(depth), int(temp*node) : int(temp*(node+1))] = lut_g[np.arange(temp).astype(int), ucapl[l], pairs[l, :, 0], pairs[l, :, 1]]
                    else:
                        if node in self.frozen_bits:
                            ucap[:, n, node] = 0
                            llr = np.zeros(self.L)
                            for l in range(self.L):
                                symbol = int(lut_g[0, ucapl[l], pairs[l, 0, 0], pairs[l, 0, 1]])
                                llr[l] = self.virtual_channel_llr[-1, node, symbol]
                            PML += np.abs(llr) * (llr < 0)
                        else:
                            decision = np.zeros(self.L)
                            PM2 = np.concatenate([PML, np.zeros(self.L)])
                            for l in range(self.L):
                                symbol = int(lut_g[0, ucapl[l], pairs[l, 0, 0], pairs[l, 0, 1]])
                                llr = self.virtual_channel_llr[-1, node, symbol]
                                decision[l] = llr < 0
                                PM2[self.L + l] = PML[l] + np.abs(llr)
                            PML, posi = self.mink(PM2, self.L)
                            posi1 = posi >= self.L
                            posi[posi1] -= self.L
                            decision = decision[posi]
                            decision[posi1] = 1 - decision[posi1]
                            symbols = symbols[posi, :, :]
                            ucap = ucap[posi, :, :]
                            ucap[:, n, node] = decision

                    node_state[node_posi] = 2

                else:
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


        # CRC-check
        idx = np.argsort(PML)
        for i in idx:
            _, check_code = crc.encode(ucap[i, n, self.msg_posi][:self.A])
            if np.all(check_code == ucap[i, n, self.msg_posi][self.A:]):
                decoded_bits = ucap[i, n, self.msg_posi][:self.A]  # extract high position bits in crc coded codewords
                return decoded_bits
        # if none of the decoded bits satisfy thr CRC check then simply choose the decoded result with smllest path metric
        idx = np.argmin(PML)
        decoded_bits = ucap[idx, n, self.msg_posi]
        return decoded_bits






