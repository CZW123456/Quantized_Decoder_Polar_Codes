import numpy as np
from ConventionalDecoder.encoder import PolarEncoder

class FastSCDecoder():

    def __init__(self, N, K, node_type, frozen_bits, msg_bits):
        self.N = N
        self.K = K
        self.frozen_bits = frozen_bits
        self.msg_posi = msg_bits
        self.node_type = node_type

    def f(self, a, b):
        return (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * np.min(np.concatenate([np.abs(a), np.abs(b)]).reshape((2, a.shape[0])), axis=0)

    def g(self, a, b, c):
        return (1 - 2 * c) * a + b

    def decode(self, channel_vector_llr, cword=None):
        n = int(np.log2(self.N))
        LLR = np.zeros((n + 1, self.N))  # LLR matrix for decoding
        LLR[0] = channel_vector_llr  # initialize the llr in the root node
        ucap = np.zeros((n + 1, self.N))  # upward decision result for reverse binary tree traversal
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
                node = node // 2
                depth -= 1
            else:  # other intermediate node
                node_posi = 2 ** depth - 1 + node  # node index in the binary tree
                if node_state[node_posi] == 0:  # 0 means this node is first achieved in the traversal, calc f value

                    # R0 node
                    if self.node_type[node_posi] == 0:
                        temp = 2 ** (n - depth)
                        ucap[depth, temp * node : temp * (node + 1)] = np.zeros(temp)
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    # R1 node
                    if self.node_type[node_posi] == 1:
                        temp = 2 ** (n - depth)
                        incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                        decision = (1 - np.sign(incoming_llr)) / 2
                        ucap[depth, temp * node: temp * (node + 1)] = decision
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue


                    # REP node
                    if self.node_type[node_posi] == 2:
                        temp = 2 ** (n - depth)
                        incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                        S = np.sum(incoming_llr)
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
                        incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                        decision = (1 - np.sign(incoming_llr)) / 2
                        decision = decision.astype(np.bool)
                        parity_check = np.mod(np.sum(decision), 2)
                        if parity_check == 0:
                            ucap[depth, temp * node: temp * (node + 1)] = decision
                        else:
                            min_abs_llr_idx = np.argmin(np.abs(incoming_llr))
                            decision[min_abs_llr_idx] = np.mod(decision[min_abs_llr_idx] + 1, 2)  # filp the bit with minimun absolute LLR
                            ucap[depth, temp * node: temp * (node + 1)] = decision
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue


                    # Type I node
                    if self.node_type[node_posi] == 4:
                        temp = 2 ** (n - depth)
                        #print("Type I:{:d}".format(temp))
                        incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                        x1 = (1 - np.sign(np.sum(incoming_llr[0:temp:2]))) / 2
                        x0 = (1 - np.sign(np.sum(incoming_llr[1:temp:2]))) / 2
                        ucap[depth, temp * node : temp * (node + 1)] = np.array([[x1, x0]]).reshape((2, 1)).repeat(temp//2, 1).transpose().reshape(1, temp)
                        # return to its parent node
                        node = node // 2
                        depth -= 1
                        continue


                    # Type II node
                    if self.node_type[node_posi] == 5:
                        temp = 2 ** (n - depth)
                        #print("Type II:{:d}".format(temp))
                        incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                        spc_input = np.sum(incoming_llr.reshape((temp // 4, 4)), axis=0)
                        spc_decision = (1 - np.sign(spc_input)) / 2
                        s = np.mod(np.sum(spc_decision), 2)
                        if s == 0:
                            ucap[depth, temp * node: temp * (node + 1)] = np.expand_dims(spc_decision, 1).repeat(temp//4, 1).transpose().reshape(temp)
                        else:
                            min_abs_llr_idx = np.argmin(np.abs(spc_input))
                            spc_decision[min_abs_llr_idx] = 1 - spc_decision[min_abs_llr_idx]
                            ucap[depth, temp * node: temp * (node + 1)] = np.expand_dims(spc_decision, 1).repeat(temp//4, 1).transpose().reshape(temp)


                    # Type III node
                    if self.node_type[node_posi] == 6:
                        temp = 2 ** (n - depth)
                        #print("Type III:{:d}".format(temp))
                        incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                        even_bits = incoming_llr[0:temp:2]
                        odd_bits = incoming_llr[1:temp:2]
                        spc_input = np.concatenate([even_bits, odd_bits]).reshape(2, temp//2)
                        spc_decision = (1 - np.sign(spc_input)) / 2
                        s = np.mod(np.sum(spc_decision, axis=1), 2)
                        if np.all(s == 0):
                            ucap[depth, temp * node : temp * (node + 1) : 2] = spc_decision[0]
                            ucap[depth, temp * node + 1 : temp * (node + 1) : 2] = spc_decision[1]
                        else:
                            min_abs_llr_idx = np.argmin(np.abs(spc_input[s == 1]))
                            for i in range(2):
                                if s[i] == 1:
                                    min_abs_llr_idx = np.argmin(np.abs(spc_input[i]))
                                    spc_decision[i, min_abs_llr_idx] = 1 - spc_decision[i, min_abs_llr_idx]
                            ucap[depth, temp * node: temp * (node + 1): 2] = spc_decision[0]
                            ucap[depth, temp * node: temp * (node + 1): 2] = spc_decision[1]


                    # Type IV node
                    if self.node_type[node_posi] == 7:
                        temp = 2 ** (n - depth)
                        #print("Type IV:{:d}".format(temp))
                        incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                        spc_input = incoming_llr.reshape((temp//4, 4))
                        z = np.arctanh(np.prod(np.tanh(spc_input / 2), axis=0)) # may be implemented in hardware friendly mode
                        z_hat = int((1 - np.sign(np.sum(z))) // 2)
                        spc_decision = (1 - np.sign(spc_input)) / 2
                        s = np.mod(np.sum(spc_decision, axis=0), 2)
                        if np.all(s == z_hat):
                            ucap[depth, temp * node: temp * (node + 1): 4] = spc_decision[:, 0]
                            ucap[depth, temp * node + 1: temp * (node + 1): 4] = spc_decision[:, 1]
                            ucap[depth, temp * node + 2: temp * (node + 1): 4] = spc_decision[:, 2]
                            ucap[depth, temp * node + 3: temp * (node + 1): 4] = spc_decision[:, 3]
                        else:
                            for i in range(temp//4):
                                if s[i] != z_hat:
                                    min_abs_llr_idx = np.argmin(np.abs(spc_input[:, i]), axis=0)
                                    spc_decision[min_abs_llr_idx, i] = 1 - spc_decision[min_abs_llr_idx, i]
                            ucap[depth, temp * node: temp * (node + 1): 4] = spc_decision[:, 0]
                            ucap[depth, temp * node + 1: temp * (node + 1): 4] = spc_decision[:, 1]
                            ucap[depth, temp * node + 2: temp * (node + 1): 4] = spc_decision[:, 2]
                            ucap[depth, temp * node + 3: temp * (node + 1): 4] = spc_decision[:, 3]


                    # Type V Node
                    if self.node_type[node_posi] == 8:
                        temp = 2 ** (n - depth)
                        #print("Type V:{:d}".format(temp))
                        incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                        y_8 = np.sum(incoming_llr.reshape((temp//8, 8)), axis=0)
                        a = y_8[:4]
                        b = y_8[4:]
                        z = int((1 - np.sign(np.sum(self.f(a, b)))) / 2)
                        spc_input = self.g(a, b, z)
                        spc_decision = (1 - np.sign(spc_input)) / 2
                        s = np.mod(np.sum(spc_decision), 2)
                        if s == 0:
                            x4_7 = spc_decision
                            x0_3 = np.mod(spc_decision + z, 2)
                            ucap[depth, temp * node: temp * (node + 1)] = np.concatenate([x0_3, x4_7]).reshape(8, 1).repeat(temp//8, 1).transpose().reshape(1, temp)
                        else:
                            min_abs_llr_idx = np.argmin(np.abs(spc_input))
                            spc_decision[min_abs_llr_idx] = 1 - spc_decision[min_abs_llr_idx]
                            x4_7 = spc_decision
                            x0_3 = np.mod(spc_decision + z, 2)
                            ucap[depth, temp * node: temp * (node + 1)] = np.concatenate([x0_3, x4_7]).reshape(8, 1).repeat(temp//8, 1).transpose().reshape(1, temp)


                    temp = 2 ** (n - depth)
                    incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                    a = incoming_llr[:temp // 2]
                    b = incoming_llr[temp // 2:]
                    # calc location for the left child
                    node *= 2
                    depth += 1
                    # length of the incoming belief vector of the left node
                    temp /= 2
                    LLR[int(depth), int(temp * node): int(temp) * int(node + 1)] = self.f(a, b)  # update the LLR in current node
                    node_state[node_posi] = 1  # switch node state from 0 to 1

                elif node_state[node_posi] == 1:  # 1 means this node is visited once, and it should visit its right child, calc g value
                    temp = 2 ** (n - depth)
                    incoming_llr = LLR[depth, temp * node: temp * (node + 1)]
                    a = incoming_llr[:temp // 2]
                    b = incoming_llr[temp // 2:]
                    ltemp = temp // 2
                    lnode = 2 * node
                    cdepth = depth + 1
                    ucapl = ucap[cdepth, ltemp * lnode: ltemp * (lnode + 1)]  # incoming decision from the left child
                    node = 2 * node + 1
                    depth += 1
                    temp /= 2
                    LLR[depth, int(temp) * node: int(temp) * (node + 1)] = self.g(a, b, ucapl)
                    node_state[node_posi] = 2  # switch node state from 1 to 2
                else:  # left and right child both have been traversed, now summarize decision from the two nodes to the parent
                    temp = 2 ** (n - depth)
                    ctemp = temp // 2
                    lnode = 2 * node
                    rnode = 2 * node + 1
                    cdepth = depth + 1
                    ucapl = ucap[cdepth, ctemp * lnode: ctemp * (lnode + 1)]
                    ucapr = ucap[cdepth, ctemp * rnode: ctemp * (rnode + 1)]
                    ucap[depth, int(temp) * node: int(temp) * (node + 1)] = np.concatenate([np.mod(ucapl + ucapr, 2), ucapr], axis=0)  # summarize function
                    if node == 0 and depth == 0:  # if this is the last bit to be decoded
                        done = True
                    else:
                        node = node // 2
                        depth -= 1

        x = ucap[0]  # obtain decoding result in the channel part, not the transmitter part
        m = 1
        # recode the decoded channel part code word to obtain the ultimate transmitted bits
        for d in range(n - 1, -1, -1):
            for i in range(0, self.N, 2*m):
                a = x[i : i + m]                                        # first part
                b = x[i + m : i + 2 * m]                                # second part
                x[i: i + 2 * m] = np.concatenate([np.mod(a + b, 2), b]) #combining
            m *= 2
        u = x[self.msg_posi]
        return u

