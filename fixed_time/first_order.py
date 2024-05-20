import numpy as np
from base.base import ABCProtocol

"""
在不同论文中，会出现参数符号不一致的情况
"""
class FisrtFixedProtocol(ABCProtocol):
    def __init__(self, alpha=1, beta=1, gamma=1.1):
        # 一致性协议自定义的参数
        self.alpha = alpha
        self.beta = beta
        # 计算时间上限的参数
        self.gamma = gamma
        self.mu = 1 - 1 / self.gamma
        self.upsilon = 1 + 1 / self.gamma
    
    def time_bound_estimate(self, A: np.array):
        # 参数
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        mu = self.mu
        ups = self.upsilon
        n = A.shape[0]
        A_mu = np.power(A, 2 * mu / (mu + 1))
        A_ups = np.power(A, 2 * ups / (ups + 1))
        D_mu = np.diag(np.sum(A_mu, axis=1))
        L_mu = D_mu - A_mu
        D_ups = np.diag(np.sum(A_ups, axis=1))
        L_ups = D_ups - A_ups
        # 特征根
        vals_mu, _ = np.linalg.eig(L_mu)
        vals_ups, _ = np.linalg.eig(L_ups)
        val2_mu = np.min(vals_mu[vals_mu > 1e-7])  # 其会求出接近0但极小的量
        val2_ups = np.min(vals_ups[vals_ups > 1e-7])
        # 计算时间上限
        T_max_1 = 2 / (alpha * np.power(2, mu) * np.power(val2_mu, (mu + 1) / 2) * (1 - mu)) + \
                    2 / (beta * np.power(2, ups) * np.power(n, (1 - ups) / 2) * \
                    np.power(val2_ups, (ups + 1) / 2) * (ups - 1))
        T_max_2 = np.pi * gamma * np.power(n, 1 / (4 * gamma)) / (2 * np.power(alpha * beta, 1 / 2) * \
                    np.power(val2_mu, 1 / 2 - 1 / (4 * gamma)) * \
                    np.power(val2_ups, 1 / 2 + 1 / (4 * gamma)))
        print("T_max_1: %f, T_max_2: %f", T_max_1, T_max_2)
        T_max_array = np.array([T_max_1, T_max_2])
        return T_max_array
    
    def get_u(self):
        pass


if __name__ == '__main__':
    A = np.array([[0, 5, 0, 6, 0, 0],
                [5, 0, 7, 0, 4, 0],
                [0, 7, 0, 3, 0, 0],
                [6, 0, 3, 0, 2, 0],
                [0, 4, 0, 2, 0, 1],
                [0, 0, 0, 0, 1, 0]])
    ffp = FisrtFixedProtocol()
    T_max_array = ffp.time_bound_estimate(A)
