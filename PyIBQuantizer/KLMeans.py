from PyIBQuantizer.IB_base_class import GenericIB
import numpy as np
from PyIBQuantizer.inf_theory_tools import kl_divergence, mutual_information

class KLMeansIB(GenericIB):

    def __init__(self, p_x_y, card_T, nrun, symmetric_init=True):

        GenericIB.__init__(self, p_x_y, card_T, np.inf, [], nrun)
        self.name = 'KL means IB'

        # self.cost_mat= np.zeros((self.cardinality_Y,self.cardinality_Y))+np.inf

        self.symmetric_init = symmetric_init
        self.p_y = self.p_x_y.sum(1)
        self.p_t = np.zeros(self.card_T)

        # calculate p(x|y)
        self.p_x_given_y = self.p_x_y / self.p_y[:, np.newaxis]


    def KL_divergence_mat(self):
        KL_div_mat = np.zeros((self.card_Y, self.card_T))
        for c in range(self.card_T):
            KL_div_mat[:, c] = kl_divergence(self.p_x_given_y, self.p_x_given_t[c, :])
        return KL_div_mat


    def assign(self):
        '''
        compute u_k = argmax_{l} D_{KL} {p(x|y)||p(x|t_l)}
        the prob vector p(x|y) should be assign to the cluster that has the most similar
        distribution with it
        :return:
        '''
        p_t_given_y = np.argmin(self.KL_divergence_mat(), axis=1)

        # ensure that no cluster is empty
        for t in range(self.card_T):
            indices = np.where(p_t_given_y == t)[0]
            if indices.size == 0:
                indices = self.last_resort[t]
                p_t_given_y[int(indices)] = int(t)
            else:
                self.last_resort[t] = indices[-1]

        return p_t_given_y


    def KL_means_run(self):
        """ This function tries to minimize the information bottleneck functional using a KL means_algorithm."""

        # Initialization
        # number of identity matrices fitting inside p_t_given_y
        neye = int(np.floor(self.card_Y / (self.card_T + 1)))
        # remaining rows that will be filled up with ones in the first row
        remainder = int((self.card_Y - neye * self.card_T))

        # preallocate arrays
        ib_fct = np.zeros(self.nrun)
        I_TX = np.zeros(self.nrun)
        counter_vec = np.zeros(self.nrun)
        p_t_given_y_mats = np.zeros((self.card_Y, self.card_T, self.nrun))
        p_t_mats = np.zeros((1, self.card_T, self.nrun))
        p_x_given_t_mats = np.zeros((self.card_T, self.card_X, self.nrun))

        # run for-loop for each number of run
        for run in range(0, self.nrun):

            # Begin initialization
            self.p_t_given_y = np.zeros((self.card_Y, self.card_T + 1))
            self.p_t_given_y[:int(neye * self.card_T), :self.card_T] = np.tile(np.eye(self.card_T), (neye, 1))
            self.p_t_given_y[self.card_Y - remainder:, 0] = np.ones(remainder)
            # assign a prob vector u_y = P(x|y) to a cluster v_k = P(x|t_k) randomly
            self.p_t_given_y = self.p_t_given_y[np.random.permutation(self.card_Y), :]

            p_t_given_y = np.argmax(self.p_t_given_y, axis=1)

            self.last_resort = np.zeros(self.card_T)  # these vector has to ensure that at least one entry is in one cluster
            for t in range(self.card_T):
                indices = np.where(p_t_given_y == t)[0]
                # grab one entry from each cluster
                smallest_contribution = np.argmin(self.p_y[indices])
                self.last_resort[t] = indices[smallest_contribution]
                # calculate p(t)
                self.p_t[t] = self.p_y[indices].sum(0)
                # calculate p(x|t)

                self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[indices, :].sum(0)

            p_xt = self.p_x_given_t[:self.card_T, :] * self.p_t[:, np.newaxis]
            old_MI = mutual_information(p_xt)
            new_MI = 0

            # Processing
            counter = 0
            # repeat until stable solution found
            while np.abs(old_MI - new_MI) > 1e-11 and counter < self.card_T * 10:
                counter += 1
                old_MI = new_MI

                # estimation step
                p_t_given_y = self.assign()

                # update step
                for t in range(self.card_T):
                    indices = np.where(p_t_given_y == t)[0]
                    if indices.size == 0:
                        indices = self.last_resort[t]
                        p_t_given_y[int(indices)] = int(t)
                        self.p_t[t] = self.p_y[int(indices)]
                        self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[int(indices), :]
                    else:
                        # grab one entry from each cluster
                        self.last_resort[t] = indices[-1]
                        # calculate p(t)
                        self.p_t[t] = self.p_y[indices].sum(0)
                        # calculate p(x|t)
                        self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[indices, :].sum(0)

                p_xt = self.p_x_given_t * self.p_t[:, np.newaxis]
                new_MI = mutual_information(p_xt)
                print("run = {:d}, iter = {:d}, MI = {:f}".format(run, counter, new_MI))

            self.p_t_given_y = np.zeros((self.card_Y, self.card_T))
            for i in range(self.card_T):
                self.p_t_given_y[p_t_given_y == i, i] = 1
            counter_vec[run] = counter
            p_t_given_y_mats[:, :, run] = self.p_t_given_y
            p_t_mats[:, :, run] = self.p_t
            p_x_given_t_mats[:, :, run] = self.p_x_given_t

            p_xt = self.p_x_given_t[:self.card_T, :] * self.p_t[:, np.newaxis]
            p_xt = p_xt / p_xt.sum()

            I_TX[run] = mutual_information(p_xt)
            ib_fct[run] = I_TX[run]

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)
        print('Winner finished in ', counter_vec[winner], ' iterations.')
        print('Average number of iterations to finished:', np.mean(counter_vec), )
        self.p_t_given_y = p_t_given_y_mats[:, :, winner].squeeze()
        self.p_x_given_t = p_x_given_t_mats[:, :, winner].squeeze()
        self.p_t = p_t_mats[:, :, winner].squeeze()
        self.MI_XY = mutual_information(self.p_x_y)
        self.MI_XT = I_TX[winner]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    card_Y = 2000
    card_T = 8
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

    quantizer = KLMeansIB(p_y_and_x.transpose(), card_T, 5)
    quantizer.KL_means_run()
    p_t_given_y, p_x_given_t, p_t = quantizer.get_result()
    quantizer.display_MIs("KL Means IB")
    lut = np.argmax(p_t_given_y, axis=1)

    plt.stem(np.arange(card_T), np.log(p_x_given_t[:, 0] / p_x_given_t[:, 1]), use_line_collection=True)
    plt.title("Meaning p(x|t) as LLRs")
    plt.xlabel("t")
    plt.ylabel("L(x|t)")
    plt.show()
