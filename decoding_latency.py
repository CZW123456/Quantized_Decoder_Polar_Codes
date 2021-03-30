import numpy as np
from PyIBQuantizer.inf_theory_tools import mutual_information
import argparse
from ConventionalDecoder.CodeConstruction import PolarCodeConstructor
from ConventionalDecoder.IdentifyNodes import NodeIdentifier
from QuantizeDecoder.QDensityEvolution_MMI import channel_transition_probability_table
from quantizers.quantizer.MMI import MMIQuantizer
import os
import pickle as pkl

class DecodingLatencyCalculator():

    def __init__(self, N, K, L, frozen_bits, message_bits, node_type):
        self.N = N
        self.K = K
        self.L = L
        self.frozen_bits = frozen_bits
        self.msg_bits = message_bits
        self.node_type = node_type

    def run_fast(self):
        n = int(np.log2(self.N))
        node_state = np.zeros(self.N - 1)       # node state
        depth = 0
        node = 0
        done = False
        decoding_latency = 0
        while done == False:
            if depth == n:
                decoding_latency += 1
                node = node // 2
                depth -= 1
            else:
                node_posi = int(2 ** depth - 1 + node)  # node index in the binary tree

                if node_state[node_posi] == 0:

                    # R0 node
                    if self.node_type[node_posi] == 0:
                        decoding_latency += 1
                        node = node // 2
                        depth -= 1
                        continue

                    # R1 node
                    if self.node_type[node_posi] == 1:
                        temp = 2 ** (n - depth)
                        decoding_latency += np.min([temp, self.L - 1])
                        node = node // 2
                        depth -= 1
                        continue

                    # REP node
                    if self.node_type[node_posi] == 2:
                        decoding_latency += 1
                        node = node // 2
                        depth -= 1
                        continue

                    # SPC node
                    if self.node_type[node_posi] == 3:
                        temp = 2 ** (n - depth)
                        decoding_latency += np.min([temp, self.L - 1])
                        node = node // 2
                        depth -= 1
                        continue

                    decoding_latency += 1
                    node *= 2
                    depth += 1
                    node_state[node_posi] = 1

                elif node_state[node_posi] == 1:
                    decoding_latency += 1
                    node = 2 * node + 1
                    depth += 1
                    node_state[node_posi] = 2

                else:
                    if node == 0 and depth == 0:
                        done = True
                    else:
                        node = node // 2
                        depth -= 1

        return decoding_latency

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--L", type=int, default=8)
    args = parser.parse_args()

    # polar codes configuration
    N = args.N  # code length
    K = args.K  # information bits length
    L = args.L  # list size
    rate = K / N
    print("N = {:d}, K = {:d}, R = {:.2f}".format(N, K, rate))

    # code construction
    constructor = PolarCodeConstructor(N, K, "./reliable sequence.txt")
    frozenbits, msgbits, frozen_indicator, message_indicator = constructor.PW()  # PW code construction

    # node type identification
    node_identifier = NodeIdentifier(N, K, frozenbits, msgbits, use_new_node=False)
    node_type = node_identifier.run().astype(np.int32)

    calculator = DecodingLatencyCalculator(N, K, L, frozenbits, msgbits, node_type)

    latency_fast = calculator.run_fast()
    latency_regular = 2 * N - 2 + K
    ratio = 1 - latency_fast / latency_regular
    print("latency_regular = {:d}, latency_fast = {:d} ratio = {:f}".format(latency_regular, latency_fast, ratio))

