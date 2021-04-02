import numpy as np

class PolarEncoder():

    def __init__(self, N, K, frozen_bits, msg_bits):
        # read reliable sequence according to the 5G NR standard
        self.N = N
        self.K = K
        self.rate = K / N
        self.n = int(np.log2(N))
        # reliable sequence for N
        self.frozen_posi = frozen_bits
        self.msg_bits = msg_bits

    def non_system_encoding(self, bits):
        u = np.zeros(self.N).astype(np.int)
        u[self.msg_bits] = bits  # non frozen bits are set as information bits
        # begin non systematic Polar encoding
        m = 1 # number of bits combined
        for d in range(self.n - 1, -1, -1):
            for i in range(0, self.N, 2 * m):
                a = u[i : i + m]  # first part
                b = u[i + m : i + 2 * m] # second part
                u[i: i + 2 * m] = np.concatenate([np.mod(a + b, 2), b])  #combining
            m *= 2
        return u

    def systematic_encoding(self, bits):
        v = np.zeros(self.N).astype(np.int)
        v[self.msg_bits] = bits  # non frozen bits are set as information bits
        # begin non systematic Polar encoding
        m = 1  # number of bits combined
        for d in range(self.n - 1, -1, -1):
            for i in range(0, self.N, 2 * m):
                a = v[i: i + m]  # first part
                b = v[i + m: i + 2 * m]  # second part
                v[i: i + 2 * m] = np.concatenate([np.mod(a + b, 2), b])  # combining
            m *= 2
        v[self.frozen_posi] = 0
        # begin non systematic Polar encoding again
        m = 1  # number of bits combined
        for d in range(self.n - 1, -1, -1):
            for i in range(0, self.N, 2 * m):
                a = v[i: i + m]  # first part
                b = v[i + m: i + 2 * m]  # second part
                v[i: i + 2 * m] = np.concatenate([np.mod(a + b, 2), b])  # combining
            m *= 2
        return v

if __name__ == "__main__":
    N = 64
    K = 2
    frozen_bits = np.arange(0, N - 2)
    msg_bits = np.array([N - 2, N - 1])
    bits = np.array([1, 1])
    x = PolarEncoder(N, K, frozen_bits, msg_bits).non_system_encoding(bits)
    print(x)