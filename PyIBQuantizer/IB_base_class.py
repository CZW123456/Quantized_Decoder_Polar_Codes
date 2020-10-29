from PyIBQuantizer.inf_theory_tools import *

class GenericIB:


    def __init__(self, p_x_y, card_T, beta, eps, nrun):
        # input variables
        self.p_x_y = p_x_y
        self.card_T = card_T
        self.card_X = p_x_y.shape[1]
        self.card_Y = p_x_y.shape[0]
        self.beta = beta
        self.eps = eps
        self.nrun = nrun

        # initialize output variables
        self.MI_XT = 1
        self.MI_XY = 1
        self.p_t_given_y = np.zeros((self.card_Y, self.card_T))
        self.p_x_given_t = np.zeros((self.card_T, self.card_X))
        self.p_t = np.zeros(self.card_T)


    def calc_merge_cost(self):
        # find merge distribution, defined in Slomn's Ph.D thesis P34, Equation 3.11, p_t_bar is a vector in R^{card_T}
        # which contains the merge probability of merging current symbol y to other cluster
        p_t_bar = self.p_t[-1] + self.p_t[:-1] # merge probability after merging y to each cluster
        pi1 = self.p_t[-1] / p_t_bar
        pi2 = self.p_t[:-1] / p_t_bar
        # compute the distance between singleton cluster and other clusters
        d_bar = js_divergence(self.p_x_given_t[-1, :], self.p_x_given_t[:-1, :], pi1, pi2) - (pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta
        # merge cost equals to the merge distribution multiplies the distance from singleton cluster to other cluster
        cost_vec = p_t_bar * d_bar
        return cost_vec

    def display_MIs(self, name):
        print('----- Mutual Information Comp --- ')
        print('----- ', name, ' ------ ')
        print('MI_XT_s= ', str(self.MI_XT))
        print('MI_XY_s= ', str(self.MI_XY))
        print('ratio= ', str(self.MI_XT / self.MI_XY))

    def display_result(self):
        return {'p_t_given_y': self.p_t_given_y,
                'p_x_given_t': self.p_x_given_t,
                'p_t': self.p_t,
                'MI_XT': self.MI_XT,
                'MI_XY': self.MI_XY}

    def get_result(self):
        return self.p_t_given_y, self.p_x_given_t, self.p_t



