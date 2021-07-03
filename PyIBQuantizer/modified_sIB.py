import numpy as np
import PyIBQuantizer.inf_theory_tools as inf_tool


class modified_sIB():

    def __init__(self, card_T_, beta=1e30, nror_=5):
        '''
        This class implements the modified sequential IB algorithm:
        (1) segment interval [0, card_Y] to card_T intervals such that the interval borders are increasing, e.g. [0, 233, 346, .., 2000]
        (2) until a stable solution is found, do:
            1. search the border vector defined above in scan order, try to merge the last element in each cluster to its neighbor cluster according to the same cost
               as sequential IB algorithm
            2. search the border vector defined above in reverse scan order, try to merge the first element in each cluster to its neighbor cluster according to the same cost
               as sequential IB algorithm
        Key distinction between this modified sequential IB algorithm and the basic sequential IB algorithm is that it can obtain an ordered LUT which means the cluster border
        in the result LUT is monotonic increasing. What's more, when computing merge cost, only 2 cluster are taken into consideration, which significantly reduces the computation overhead
        :param p_x_y_:
        :param card_T_:
        :param nror_:
        '''
        self.card_T = card_T_
        self.nrun = nror_
        self.beta = beta
        self.max_run = 1000
        self.name = 'modified sIB'

    def calc_merge_cost(self, p_t, p_x_given_t, border_between_clusters, cur_card_T):
        bbc = border_between_clusters
        cur_card_T = cur_card_T - 1
        # p_t means the cluster probability of merging the last element to its neighbor cluster forming a new cluster or
        # staying in current cluster
        p_t_bar = p_t[cur_card_T] + p_t[[bbc, bbc + 1]]
        pi1 = np.zeros_like(p_t_bar)
        pi2 = np.zeros_like(p_t_bar)
        pi1[p_t_bar != 0] = p_t[cur_card_T] / p_t_bar[p_t_bar != 0]
        if p_t_bar[0] != 0:
            pi2[0] = p_t[bbc] / p_t_bar[0]
        if p_t_bar[1] != 0:
            pi2[1] = p_t[bbc + 1] / p_t_bar[1]
        cost_vec = p_t_bar * inf_tool.js_divergence(p_x_given_t[cur_card_T, :], p_x_given_t[[bbc, bbc + 1], :], pi1, pi2)
        return cost_vec

    def modified_sIB_run(self, pxy, border_vectors):
        self.p_x_y = pxy
        llr = np.zeros(pxy.shape[0])
        llr[pxy[:, 1] == 0] = 1e300
        llr[pxy[:, 1] != 0] = inf_tool.log2_stable(pxy[pxy[:, 1] != 0, 0]) - inf_tool.log2_stable((pxy[pxy[:, 1] != 0, 1]))
        permutation = np.argsort(llr)

        self.p_x_y = self.p_x_y[permutation, :]

        # set static values
        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        cardinality_X = p_x.shape[0]
        cardinality_Y = p_y.shape[0]

        cur_card_T = self.card_T

        # preallocate arrays
        ib_fct = np.zeros(self.nrun)
        I_YT = np.zeros(self.nrun)
        I_TX = np.zeros(self.nrun)
        p_t_given_y_mats = np.zeros((cardinality_Y, self.card_T + 1, self.nrun))

        flag = False

        # run for-loop for each number of run
        for run in range(self.nrun):
            p_t_given_y = np.zeros((cardinality_Y, self.card_T + 1))
            border_vec = border_vectors[run]
            a = 0
            for t in range(0, self.card_T):
                p_t_given_y[a:border_vec[t], t] = 1
                a = border_vec[t]
            # Processing
            init_mat = p_t_given_y
            end_mat = np.zeros((cardinality_Y, self.card_T + 1))
            # repeat until stable solution found
            num_run = 0
            while not np.array_equal(init_mat, end_mat) and num_run < self.max_run:
                last_cluster_vec = np.hstack([np.zeros(self.card_T), 1])
                init_mat = np.copy(p_t_given_y)
                # modify the cluster border in scan-order and reverse scan order
                for border_between_clusters in range(0, self.card_T - 1):
                    done_left_to_right = False
                    done_right_to_left = False

                    while not done_left_to_right:
                        done_left_to_right = True
                        # find last element in the cluster
                        # this is a trick here because argmax returns first hit so flipping the array first.
                        last_elem = p_t_given_y.shape[0] - np.argmax(p_t_given_y[::-1, border_between_clusters] > 0) - 1
                        old_cluster = border_between_clusters
                        # if old cluster is not empty
                        if np.sum(p_t_given_y[:, old_cluster]) > 1:
                            # set the last element in current cluster as a singleton cluster
                            p_t_given_y[last_elem, :] = last_cluster_vec
                            cur_card_T += 1
                            # calculate p(t)  new
                            p_t = (p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)
                            # calculate p(x | t) new
                            p_x_and_t = np.dot(p_t_given_y[:, :cur_card_T].T, self.p_x_y)
                            p_x_given_t = np.zeros_like(p_x_and_t)
                            p_t_tmp = p_t[:cur_card_T]
                            p_x_given_t[p_t_tmp != 0, 0] = p_x_and_t[p_t_tmp != 0, 0] / p_t_tmp[p_t_tmp != 0]
                            p_x_given_t[p_t_tmp != 0, 1] = p_x_and_t[p_t_tmp != 0, 1] / p_t_tmp[p_t_tmp != 0]
                            # p_x_and_t = np.dot(p_t_given_y[:, :cur_card_T].T, self.p_x_y)
                            # p_x_given_t = p_x_and_t / p_t[:cur_card_T, np.newaxis]
                            # calculate merge cost of merging last element in current cluster to its neighbor cluster
                            merger_costs = self.calc_merge_cost(p_t, p_x_given_t, border_between_clusters, cur_card_T)
                            ind_min = np.argmin(merger_costs)
                            # ind_min = 0 means staying unchanged for current cluster obtains less cost
                            if ind_min == 0 and np.abs(merger_costs[0] - merger_costs[1]) > 1e-9:
                                p_t_given_y[last_elem, border_between_clusters] = 1
                            else:
                                p_t_given_y[last_elem, border_between_clusters + 1] = 1
                                done_left_to_right = False
                            p_t_given_y[last_elem, cur_card_T - 1] = 0
                            cur_card_T -= 1
                    # check other direction
                    while not done_right_to_left:
                        done_right_to_left = True
                        # find first element in the cluster
                        first_elem = np.argmax(p_t_given_y[:, border_between_clusters + 1] > 0)
                        old_cluster = border_between_clusters + 1

                        if np.sum(p_t_given_y[:, old_cluster]) > 1:
                            p_t_given_y[first_elem, :] = last_cluster_vec
                            cur_card_T += 1
                            # calculate p(t)  new
                            p_t = (p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)
                            # calculate p(x | t) new
                            p_x_and_t = np.dot(p_t_given_y[:, :cur_card_T].T, self.p_x_y)
                            p_x_given_t = np.zeros_like(p_x_and_t)
                            p_t_tmp = p_t[:cur_card_T]
                            p_x_given_t[p_t_tmp != 0, 0] = p_x_and_t[p_t_tmp != 0, 0] / p_t_tmp[p_t_tmp != 0]
                            p_x_given_t[p_t_tmp != 0, 1] = p_x_and_t[p_t_tmp != 0, 1] / p_t_tmp[p_t_tmp != 0]
                            # p_x_and_t = np.dot(p_t_given_y[:, :cur_card_T].T, self.p_x_y)
                            # p_x_given_t = p_x_and_t / p_t[:cur_card_T, np.newaxis]
                            merger_costs = self.calc_merge_cost(p_t, p_x_given_t, border_between_clusters, cur_card_T)
                            ind_min = np.argmin(merger_costs)
                            # ind_min = 0 means staying unchanged for current cluster obtains less cost
                            if ind_min == 0 and np.abs(merger_costs[0] - merger_costs[1]) > 1e-9:
                                p_t_given_y[first_elem, border_between_clusters] = 1
                                done_right_to_left = False
                            else:
                                p_t_given_y[first_elem, border_between_clusters + 1] = 1
                            p_t_given_y[first_elem, cur_card_T - 1] = 0
                            cur_card_T -= 1
                end_mat = p_t_given_y
                num_run += 1
                # print("num_run = {:d}".format(num_run))
                if num_run == self.max_run:
                    flag = True

            # calculate p(t)  new
            p_t = (p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)
            # calculate p(x | t) new
            p_x_given_t = 1 / (p_t[:cur_card_T, np.newaxis]) * np.dot(p_t_given_y[:, :cur_card_T].T, self.p_x_y)

            p_t_given_y_mats[:, :, run] = p_t_given_y

            p_ty = p_t_given_y[:, :self.card_T] * p_y[:, np.newaxis]
            p_xt = p_x_given_t[:self.card_T, :] * p_t[:, np.newaxis]

            I_YT[run] = inf_tool.mutual_information(p_ty)
            I_TX[run] = inf_tool.mutual_information(p_xt)

            ib_fct[run] = I_YT[run] / (-self.beta) + I_TX[run]

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)
        p_t_given_y = p_t_given_y_mats[:, :, winner].squeeze()
        p_t_given_x = np.zeros((cardinality_X, self.card_T))
        MI_XY = inf_tool.mutual_information(self.p_x_y)
        MI_XT = I_TX[winner]

        # get lut from p_t_given_y
        lut = np.argmax(p_t_given_y, axis=1)
        for i in range(self.card_T):
            p_t_given_x[:, i] = np.sum(self.p_x_y[lut==i, :], axis=0).transpose() / 0.5


        return lut, permutation, p_t_given_x, MI_XY, MI_XT, flag