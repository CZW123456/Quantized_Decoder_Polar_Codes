import numpy as np
from copy import deepcopy

class CRC:
    '''
    :param info list
    :param crc_n : int, default: 32
    :param p : list 生成多项式
    :param q : list crc后得到的商
    :param check_code : list crc后得到的余数，即计算得到的校验码
    :param code : list 最终的编码
    '''
    def __init__(self, crc_n=32, crc_p=None):
        self.crc_n = crc_n
        if crc_p == None:
            # 初始化生成多项式p
            print("Use default CRC polynomial")
            loc = [32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 2, 1, 0]
            if crc_n == 8:
                loc = [8, 2, 1, 0]
            elif crc_n == 16:
                loc = [16, 15, 2, 0]
            self.p = np.zeros(crc_n + 1)
            self.p[loc] = 1
        else:
            self.p = np.zeros(crc_n + 1)
            self.p[crc_p] = 1

    def encode(self, info):
        info = info.squeeze()
        info_copy = deepcopy(info)
        times = len(info)
        n = self.crc_n + 1
        # 左移补零
        info_copy = np.concatenate([info_copy, np.zeros(self.crc_n)])
        # 除
        for i in range(times):
            if info_copy[i] == 1:
                for j in range(n):
                    info_copy[j + i] = np.mod(info_copy[j + i] + self.p[j], 2)
        # 余数
        check_code = info_copy[-self.crc_n::]
        # 生成编码
        code = np.concatenate([info, check_code])
        return code, check_code


