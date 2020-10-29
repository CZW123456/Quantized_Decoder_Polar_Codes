from PyIBQuantizer.modified_sIB import modified_sIB
import numpy as np
import pickle as pkl


def get_LUT(lut_merge, permutation, mode="f"):
    lut_merge = lut_merge.squeeze()
    if mode == "f":
        lut = np.zeros((32, 32)).astype(np.int32)
        for i in range(32):
            indices = np.where(lut_merge == i)[0]
            symbols = permutation[indices]
            y = symbols // 32
            x = symbols % 32
            lut[y, x] = i
        return lut
    else:
        lut = np.zeros((2, 32, 32)).astype(np.int32)
        for i in range(32):
            indices = np.where(lut_merge == i)[0]
            symbols = permutation[indices]
            u0 = symbols // (32 * 32)
            tmp = symbols - u0 * 32 * 32
            y = tmp // 32
            x = tmp % 32
            lut[u0, y, x] = i
        return lut

def get_border_vector(cardY, cardT, num_run):
    alpha = np.ones(int(cardT)) * 1
    border_vectors = np.ones((num_run, cardT)) * cardY
    for run in range(num_run):
        while border_vectors[run, :-1].cumsum().max() >= cardY:
            border_vectors[run] = np.floor(np.random.dirichlet(alpha, 1) * (cardY))
            border_vectors[run, border_vectors[run] == 0] = 1
        border_vectors[run] = np.hstack([border_vectors[run, :-1].cumsum(), cardY]).astype(np.int)
    border_vectors = border_vectors.astype(np.int32)
    return border_vectors

Q = modified_sIB(32, nror_=1)
with open("error_input.pkl", "rb") as f:
    P_in = pkl.load(f)

border_vector = get_border_vector(1024, 32, 1)

lut_, permutation, p_t_given_x, MI_XY, MI_XT, flag = Q.modified_sIB_run(0.5*P_in.T, border_vector)
lut = get_LUT(lut_, permutation, mode="f")
print()

