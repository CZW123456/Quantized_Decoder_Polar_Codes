import PyIBQuantizer.inf_theory_tools as inf_tool
import numpy as np
from PyIBQuantizer.IB_base_class import GenericIB

class sIB(GenericIB):

    def __init__(self, p_x_y_, card_T_, nror_):
        GenericIB.__init__(self, p_x_y_, card_T_, np.inf, [], nror_)
        self.name = "Sequential IB"

    def sIB_run(self):

        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        cardinality_X = p_x.shape[0]
        cardinality_Y = p_y.shape[0]

        cur_card_T = self.card_T

        # Initialization
        # number of identity matrices fitting inside p_t_givem_y
        neye = int(np.floor(cardinality_Y / (self.card_T + 1)))
        # remaining rows that will be filled up with ones in the first row
        remainder = int((cardinality_Y - neye * self.card_T))

        # preallocate arrays
        ib_fct = np.zeros(self.nrun)
        I_YT = np.zeros(self.nrun)
        I_TX = np.zeros(self.nrun)
        p_t_given_y_mats = np.zeros((cardinality_Y, self.card_T + 1, self.nrun))
        p_t_mats = np.zeros((1, self.card_T, self.nrun))
        p_x_given_t_mats = np.zeros((self.card_T, cardinality_X, self.nrun))

        # run for-loop for each number of run
        for run in range(0, self.nrun):

            # random assign each y to a deterministic z
            self.p_t_given_y = np.zeros((cardinality_Y, self.card_T + 1), dtype=int)
            self.p_t_given_y[:int(neye * self.card_T), :self.card_T] = np.tile(np.eye(self.card_T), (neye, 1))

            self.p_t_given_y[cardinality_Y - remainder:, 0] = np.ones(remainder)
            self.p_t_given_y = self.p_t_given_y[np.random.permutation(cardinality_Y), :]

            # Processing
            init_mat = self.p_t_given_y
            end_mat = np.zeros((cardinality_Y, self.card_T + 1), dtype=int)

            # repeat until stable solution found
            while not np.array_equal(init_mat, end_mat):
                self.p_t = (self.p_t_given_y * p_y[:, np.newaxis]).sum(0)
                init_mat = np.copy(self.p_t_given_y)

                '''
                For every observable symbol y, try excluding it from its current cluster and merging y to other cluster.
                Each merge will generate a merge cost defined in Slomn's PhD thesis. Select the cluster with minimum merge
                cost and merge t to that cluster
                '''
                for i in range(0, cardinality_Y):
                    old_cluster = np.argmax(self.p_t_given_y[i, :])
                    # if current cluster for y is not empty
                    if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                        # delete y from its old cluster
                        self.p_t_given_y[i, old_cluster] = 0
                        self.p_t_given_y[i, -1] = 1

                        cur_card_T += 1

                        # calculate p(t) new
                        # only needs to be updated in the following columns: old_cluster, last cluster,
                        # previous new cluster
                        # special dot test
                        # p(t_bar) = p(t_i) + p(t_j)

                        # self.p_t[i, -1] = p_y[i]: current symbol y now forms a cluster which contains only itself
                        self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)
                        self.p_x_and_t = np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                        self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * self.p_x_and_t

                        merger_costs_vec = self.calc_merge_cost()

                        ind_min = np.argmin(merger_costs_vec)
                        self.p_t_given_y[i, ind_min] = 1
                        self.p_t_given_y[i, -1] = 0

                        cur_card_T -= 1

                end_mat = self.p_t_given_y

            # calculate p(t)  new
            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

            # calculate p(x | t) new
            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

            p_t_given_y_mats[:, :, run] = self.p_t_given_y
            p_t_mats[:, :, run] = self.p_t
            p_x_given_t_mats[:, :, run] = self.p_x_given_t

            p_ty = self.p_t_given_y[:, :self.card_T] * p_y[:, np.newaxis]
            p_xt = self.p_x_given_t[:self.card_T, :] * self.p_t[:, np.newaxis]

            I_YT[run] = inf_tool.mutual_information(p_ty)
            I_TX[run] = inf_tool.mutual_information(p_xt)

            ib_fct[run] = I_YT[run] / (-self.beta) + I_TX[run]

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)
        self.p_t_given_y = p_t_given_y_mats[:, :, winner].squeeze()
        self.p_x_given_t = p_x_given_t_mats[:, :, winner].squeeze()
        self.p_t = p_t_mats[:, :, winner].squeeze()
        self.MI_XY = inf_tool.mutual_information(self.p_x_y)
        self.MI_XT = I_TX[winner]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    card_Y = 2000
    card_T = 16
    mu1 = 1
    mu2 = -1
    n_sigma = 100
    sigma = 0.1
    sigmas = np.linspace(0.1, 0.6, n_sigma)
    amplitudes = np.array([-1, +1])
    p_x = np.array([0.5, 0.5])
    Qs = []

    y = np.linspace(norm.ppf(1e-12, loc=amplitudes.min(), scale=np.sqrt(sigma)),
                    norm.ppf(1 - 1e-12, loc=amplitudes.max(), scale=np.sqrt(sigma)), card_Y)  # define the eventspace
    delta_y = np.abs(y[1] - y[0])

    p_y_given_x = np.zeros((2, y.shape[0]))
    for x_idx, x in enumerate(amplitudes):
        p_y_given_x[x_idx, :] = norm.pdf(y, loc=x, scale=np.sqrt(sigma)) * delta_y

    p_y_and_x = p_y_given_x * np.expand_dims(p_x, axis=1).repeat(card_Y, axis=1)
    p_y_and_x /= p_y_and_x.sum()

    quantizer = sIB(p_y_and_x.transpose(), card_T, 5)
    quantizer.sIB_run()
    p_t_given_y, p_x_given_t, p_t = quantizer.get_result()
    quantizer.display_MIs("Sequential IB")
    lut = np.argmax(p_t_given_y, axis=1)

    plt.stem(np.arange(card_T), np.log(p_x_given_t[:, 0] / p_x_given_t[:, 1]), use_line_collection=True)
    plt.title("Meaning p(x|t) as LLRs")
    plt.xlabel("t")
    plt.ylabel("L(x|t)")
    plt.show()