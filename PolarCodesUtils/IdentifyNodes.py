import numpy as np
#from graphviz import Digraph
import os

class NodeIdentifier():
    def __init__(self, N, K, frozen_bits, msg_bits, use_new_node=True):
        self.N = N
        self.K = K
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
        n = np.log2(self.N)
        depth = 0
        node = 0
        done = False
        while done == False:
            if depth == n:                          # if leaf node
                node_pois = 2 ** depth - 1 + node
                # check the type of the leaf node
                if node in self.frozen_bits_indices:
                    node_type[node_pois] = 0
                else:
                    node_type[node_pois] = 1
                node = node // 2
                depth -= 1
            else:
                node_pois = 2 ** depth - 1 + node

                if node_state[node_pois] == 0:
                    temp = int(2 ** (n - depth))
                    constitute_code_type = inforbits_type[temp * node : temp * (node + 1)]

                    # R0 node: constitude code are all frozen bits
                    if np.sum(constitute_code_type) == 0:
                        node_type[node_pois] = 0
                        node = node // 2
                        depth -= 1
                        if len(constitute_code_type) == self.N:
                            done = True
                        continue

                    # R1 node: constitude code are all message bits
                    if np.sum(constitute_code_type) == temp:
                        node_type[node_pois] = 1
                        node = node // 2
                        depth -= 1
                        if len(constitute_code_type) == self.N:
                            done = True
                        continue

                    # REP node: constitude code are all frozen bits except for the last bits
                    if np.sum(constitute_code_type) == 1 and constitute_code_type[-1] == 1:
                        node_type[node_pois] = 2
                        node = node // 2
                        depth -= 1
                        if len(constitute_code_type) == self.N:
                            done = True
                        continue

                    # SPC node: constitude code are all message bits except for the first one
                    if np.sum(constitute_code_type) == temp - 1 and constitute_code_type[0] == 0:
                        node_type[node_pois] = 3
                        node = node // 2
                        depth -= 1
                        if len(constitute_code_type) == self.N:
                            done = True
                        continue

                    if self.use_new_node:
                        # Type I node: only last two bits are information bits, R >= 4
                        if np.sum(constitute_code_type) == 2 and constitute_code_type[-1] == 1 and constitute_code_type[-2] == 1 and temp >= 4:
                            node_type[node_pois] = 4
                            node = node // 2
                            depth -= 1
                            if len(constitute_code_type) == self.N:
                                done = True
                            continue

                        # Type II node: only last three bits are information bits, R >= 4
                        if np.sum(constitute_code_type) == 3 and constitute_code_type[-1] == 1 and constitute_code_type[
                            -2] == 1 and constitute_code_type[-3] == 1 and temp >= 4:
                            node_type[node_pois] = 5
                            node = node // 2
                            depth -= 1
                            if len(constitute_code_type) == self.N:
                                done = True
                            continue

                        # Type III node: only first two bits are frozen bits, R >= 4
                        if np.sum(constitute_code_type) == temp - 2 and constitute_code_type[0] == 0 and constitute_code_type[
                            1] == 0 and temp >= 4:
                            node_type[node_pois] = 6
                            node = node // 2
                            depth -= 1
                            if len(constitute_code_type) == self.N:
                                done = True
                            continue

                        # TypeIV node: only first three bits are frozen bits, R >= 4
                        if np.sum(constitute_code_type) == temp - 3 and constitute_code_type[0] == 0 and constitute_code_type[
                            1] == 0 and constitute_code_type[2] == 0 and  temp >= 4:
                            node_type[node_pois] = 7
                            node = node // 2
                            depth -= 1
                            if len(constitute_code_type) == self.N:
                                done = True
                            continue

                        # TypeV node: only last 1, 2, 3, 5 bits are information bits, R >= 8
                        if np.sum(constitute_code_type) == 4 and constitute_code_type[-1] == 1 and constitute_code_type[-2] == 1 \
                            and constitute_code_type[-3] == 1 and constitute_code_type[-5] == 1 and temp >= 8:
                            node_type[node_pois] = 8
                            node = node // 2
                            depth -= 1
                            if len(constitute_code_type) == self.N:
                                done = True
                            continue
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
        return node_type

    def show_traverse_path(self, node_type):
        n = np.log2(self.N)
        node_state = np.zeros(2 * self.N - 1)
        depth = 0
        node = 0
        done = False
        num_node_activate = 0
        while done == False:
            if depth == n:  # if leaf node
                node_pois = 2 ** depth - 1 + node
                # check the type of the leaf node
                if node in self.frozen_bits_indices:
                    node_type[node_pois] = 0
                else:
                    node_type[node_pois] = 1
                node = node // 2
                depth -= 1
            else:  # if intermediate node
                node_pois = 2 ** depth - 1 + node
                if node_state[node_pois] == 0:  # if current intermediate node is visited first time, go to its left child

                    if node_type[node_pois] == 0:                                # R0 node
                        print("R0 Node : {:d}".format(node_pois))
                        num_node_activate += 1
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    if node_type[node_pois] == 1:                                # R1 node
                        print("R1 Node : {:d}".format(node_pois))
                        num_node_activate += 1
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    if node_type[node_pois] == 2:                                # REP node
                        print("REP Node : {:d}".format(node_pois))
                        num_node_activate += 1
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue

                    if node_type[node_pois] == 3:                                # SPC node
                        print("SPC Node : {:d}".format(node_pois))
                        num_node_activate += 1
                        # return to its parent node immediately
                        node = node // 2
                        depth -= 1
                        continue
                    if self.use_new_node:
                        if node_type[node_pois] == 4:                                # Type I node
                            print("Type I Node : {:d}".format(node_pois))
                            num_node_activate += 1
                            # return to its parent node immediately
                            node = node // 2
                            depth -= 1
                            continue

                        if node_type[node_pois] == 5:                                # Type II node
                            print("Type II Node : {:d}".format(node_pois))
                            num_node_activate += 1
                            # return to its parent node immediately
                            node = node // 2
                            depth -= 1
                            continue

                        if node_type[node_pois] == 6:                                # Type III node
                            print("Type III Node : {:d}".format(node_pois))
                            num_node_activate += 1
                            # return to its parent node immediately
                            node = node // 2
                            depth -= 1
                            continue

                        if node_type[node_pois] == 7:                                # Type IV node
                            print("Type IV Node : {:d}".format(node_pois))
                            num_node_activate += 1
                            # return to its parent node immediately
                            node = node // 2
                            depth -= 1
                            continue

                        if node_type[node_pois] == 3:                                # Type V node
                            print("Type V Node : {:d}".format(node_pois))
                            num_node_activate += 1
                            # return to its parent node immediately
                            node = node // 2
                            depth -= 1
                            continue

                    num_node_activate += 1
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

        return num_node_activate
    #
    # def draw(self, node_type):
    #     n = np.log2(self.N)
    #     node_state = np.zeros(2 * self.N - 1)
    #     depth = 0
    #     node = 0
    #     done = False
    #     num_node_activate = 0
    #     dot = Digraph(comment='Polar Decode Tree')
    #     while done == False:
    #         if depth == n:  # if leaf node
    #             node_pois = 2 ** depth - 1 + node
    #             # check the type of the leaf node
    #             if node in self.frozen_bits_indices:
    #                 node_type[node_pois] = 0
    #             else:
    #                 node_type[node_pois] = 1
    #             node = node // 2
    #             depth -= 1
    #         else:  # if intermediate node
    #             node_pois = 2 ** depth - 1 + node
    #             temp = int(2 ** (n - depth))
    #             if node_state[node_pois] == 0:  # if current intermediate node is visited first time, go to its left chil
    #                 if node_type[node_pois] == 0:                                # R0 node
    #                     dot.node(str(node_pois), "R0:{:d}".format(temp))
    #                     num_node_activate += 1
    #                     # return to its parent node immediately
    #                     node = node // 2
    #                     depth -= 1
    #                     continue
    #
    #                 if node_type[node_pois] == 1:                                # R1 node
    #                     dot.node(str(node_pois), "R1:{:d}".format(temp))
    #                     num_node_activate += 1
    #                     # return to its parent node immediately
    #                     node = node // 2
    #                     depth -= 1
    #                     continue
    #
    #                 if node_type[node_pois] == 2:                                # REP node
    #                     dot.node(str(node_pois), "REP:{:d}".format(temp))
    #                     num_node_activate += 1
    #                     # return to its parent node immediately
    #                     node = node // 2
    #                     depth -= 1
    #                     continue
    #
    #                 if node_type[node_pois] == 3:                                # SPC node
    #                     dot.node(str(node_pois), "SPC:{:d}".format(temp))
    #                     num_node_activate += 1
    #                     # return to its parent node immediately
    #                     node = node // 2
    #                     depth -= 1
    #                     continue
    #
    #                 if self.use_new_node:
    #                     if node_type[node_pois] == 4:                                # Type I node
    #                         dot.node(str(node_pois), "Type I:{:d}".format(temp))
    #                         num_node_activate += 1
    #                         # return to its parent node immediately
    #                         node = node // 2
    #                         depth -= 1
    #                         continue
    #
    #                     if node_type[node_pois] == 5:                                # Type II node
    #                         dot.node(str(node_pois), "Type II:{:d}".format(temp))
    #                         num_node_activate += 1
    #                         # return to its parent node immediately
    #                         node = node // 2
    #                         depth -= 1
    #                         continue
    #
    #                     if node_type[node_pois] == 6:                                # Type III node
    #                         dot.node(str(node_pois), "Type III:{:d}".format(temp))
    #                         num_node_activate += 1
    #                         # return to its parent node immediately
    #                         node = node // 2
    #                         depth -= 1
    #                         continue
    #
    #                     if node_type[node_pois] == 7:                                # Type IV node
    #                         dot.node(str(node_pois), "Type IV:{:d}".format(temp))
    #                         num_node_activate += 1
    #                         # return to its parent node immediately
    #                         node = node // 2
    #                         depth -= 1
    #                         continue
    #
    #                     if node_type[node_pois] == 8:                                # Type V node
    #                         dot.node(str(node_pois), "Type V:{:d}".format(temp))
    #                         num_node_activate += 1
    #                         # return to its parent node immediately
    #                         node = node // 2
    #                         depth -= 1
    #                         continue
    #
    #                 dot.node(str(node_pois), "{:d}:{:d}".format(node_pois, temp))
    #                 num_node_activate += 1
    #                 node *= 2
    #                 depth += 1
    #                 node_state[node_pois] = 1
    #             elif node_state[node_pois] == 1:
    #                 node = 2 * node + 1
    #                 depth += 1
    #                 node_state[node_pois] = 2
    #             else:
    #                 lnode = node * 2
    #                 rnode = node * 2 + 1
    #                 cdepth = depth + 1
    #                 lnode_posi = 2 ** cdepth - 1 + lnode
    #                 rnode_posi = 2 ** cdepth - 1 + rnode
    #                 dot.edge(str(node_pois), str(lnode_posi))
    #                 dot.edge(str(node_pois), str(rnode_posi))
    #
    #                 if node_pois == 0:
    #                     done = True
    #                 else:
    #                     node = node // 2
    #                     depth -= 1
    #     dot.view(filename="Polar Decode Tree——({:d}, {:d})——UseNewNode={}".format(self.N, self.K, str(bool(self.use_new_node))), directory=os.getcwd())
    #     return num_node_activate

