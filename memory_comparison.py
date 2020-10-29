import numpy as np
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
import argparse


class NodeIdentifierCustom():
    def __init__(self, N, K, Q, frozen_bits, msg_bits, use_new_node=False):
        self.N = N
        self.K = K
        self.Q = Q
        self.nbits = int(np.log2(Q))
        self.frozen_bits_indices = frozen_bits
        self.msg_bits_indices = msg_bits
        self.use_new_node = use_new_node

    def run(self):
        '''
        identify basic nodes in the decoding tree, currently 4 basic types of nodes are supported:
        (1) rate-0 node: 0
        (2) rate-1 node: 1
        (3) repetition node: 2
        (4) single parity check node: 3
        (5) TypeI node: 41
        (6) TypeII node: 5
        (7) TypeIII node: 6
        (8) TypeIV node: 7
        (9) TypeV node: 8
        :return: node_type
        '''
        node_type = -1 * np.ones(2 * self.N - 1)  # node type for nodes in the decode tree
        node_state = np.zeros(2 * self.N - 1)     # node state vector, record current state of node
        inforbits_type = np.zeros(self.N)         # information bits type vector,  0 -> frozen bits, 1 -> message bits
        inforbits_type[self.msg_bits_indices] = 1
        total_memory = 0
        n_inter_node = 0
        n_leaf_node = 0
        n_node = 0
        n = np.log2(self.N)
        depth = 0
        node = 0
        done = False
        while done == False:
            if depth == n:                          # if leaf node
                total_memory += self.Q * self.nbits
                node_pois = 2 ** depth - 1 + node
                if node in self.frozen_bits_indices:
                    node_type[node_pois] = 0
                else:
                    node_type[node_pois] = 1
                node = node // 2
                depth -= 1
            else:
                node_pois = 2 ** depth - 1 + node
                if node_state[node_pois] == 0:
                    n_node += 1
                    total_memory += 3 * (self.Q**2)*self.nbits
                    temp = int(2 ** (n - depth))
                    constitute_code_type = inforbits_type[temp * node : temp * (node + 1)]

                    # R0 node: constitude code are all frozen bits
                    if np.sum(constitute_code_type) == 0:
                        total_memory += temp * self.Q * self.nbits
                        n_leaf_node += 1
                        node_type[node_pois] = 0
                        node = node // 2
                        depth -= 1
                        if len(constitute_code_type) == self.N:
                            done = True
                        continue

                    # R1 node: constitude code are all message bits
                    if np.sum(constitute_code_type) == temp:
                        total_memory += temp * self.Q * self.nbits
                        n_leaf_node += 1
                        node_type[node_pois] = 1
                        node = node // 2
                        depth -= 1
                        if len(constitute_code_type) == self.N:
                            done = True
                        continue

                    # REP node: constitude code are all frozen bits except for the last bits
                    if np.sum(constitute_code_type) == 1 and constitute_code_type[-1] == 1:
                        total_memory += temp * self.Q * self.nbits
                        n_leaf_node += 1
                        node_type[node_pois] = 2
                        node = node // 2
                        depth -= 1
                        if len(constitute_code_type) == self.N:
                            done = True
                        continue

                    # SPC node: constitude code are all message bits except for the first one
                    if np.sum(constitute_code_type) == temp - 1 and constitute_code_type[0] == 0:
                        total_memory += temp * self.Q * self.nbits
                        n_leaf_node += 1
                        node_type[node_pois] = 3
                        node = node // 2
                        depth -= 1
                        if len(constitute_code_type) == self.N:
                            done = True
                        continue

                    n_inter_node += 1
                    node *= 2
                    depth += 1
                    node_state[node_pois] = 1
                elif node_state[node_pois] == 1:
                    node = 2 * node + 1
                    depth += 1
                    node_state[node_pois] = 2
                else:
                    if node_pois == 0:
                        done = True
                    else:
                        node = node // 2
                        depth -= 1
        return node_type, total_memory, n_leaf_node, n_inter_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--Q", type=int, default=32)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    Q = args.Q  # quantize level
    rate = K / N
    # initialize encoder and decoder

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozenbits_indicator, messagebits_indicator = constructor.PW()  # PW code construction

    # node type identification
    node_identifier = NodeIdentifierCustom(N, K, Q, frozenbits, msgbits, use_new_node=False)
    node_type, total_momory, n_leaf, n_inter = node_identifier.run()
    print(n_leaf + n_inter)
    print(n_leaf)
    print(n_inter)
    print(total_momory)
    print(int(3*(N-1)*(Q**2)*int(np.log2(Q))+N*Q*np.log2(Q)))
    print(int(n_inter*3*(Q**2)*int(np.log2(Q))+N*Q*np.log2(Q)))
    print(total_momory / int(3*(N-1)*(Q**2)*int(np.log2(Q))+N*Q*np.log2(Q)))

