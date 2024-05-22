import numpy as np
from base.base import ABCProtocol, FirstOrderSystem
from base.utils import show_result, sign_abs

"""
"A new class of finite time nonlinear consensus protocols for multi agent systems"
"""
class FisrtFixedProtocol(ABCProtocol):
    def __init__(self, A, alpha=2, beta=2, p=7, q=9):
        self.A = A
        # 一致性协议自定义的参数
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
    
    def time_bound_estimate(self):
        pass
    
    def get_u(self, x):
        A = self.A
        alpha = self.alpha
        beta = self.beta
        p = self.p
        q = self.q
        u = np.zeros_like(x, dtype=np.float64)
        u_temp = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                u_temp += alpha * A[i][j] * sign_abs(x[j]-x[i], 2-p/q) + \
                        beta * A[i][j] * sign_abs(x[j]-x[i], p/q)
                # u_temp += A[i][j] * (x[j]-x[i])  # only for test
            u[i] = u_temp
            u_temp = 0
        return u


"""
"Fixed-time_consensus_algorithm_for_multi-agent_sys"
NOTE:
    1. 未收敛
"""
class FisrtFixedProtocol_1(ABCProtocol):
    def __init__(self, A, alpha=1, beta=1, gamma=1.1):
        self.A = A
        # 一致性协议自定义的参数
        self.alpha = alpha
        self.beta = beta
        # 计算时间上限的参数
        self.gamma = gamma
        self.mu = 1 - 1 / self.gamma
        self.upsilon = 1 + 1 / self.gamma
    
    def time_bound_estimate(self):
        A = self.A
        # 参数
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        mu = self.mu
        ups = self.upsilon
        n = A.shape[0]
        # A_mu = np.power(A, 2 * mu / (mu + 1))
        # A_ups = np.power(A, 2 * ups / (ups + 1))
        A_mu = sign_abs(A, 2 * mu / (mu + 1))
        A_ups = sign_abs(A, 2 * ups / (ups + 1))
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
    
    def get_u(self, x):
        P = self.get_p(x)
        D_p = np.diag(np.sum(P, axis=1))
        L_p = D_p - P
        u = -np.dot(L_p, x)
        return u
    
    def get_p(self, x):
        # P -> Phi
        A = self.A
        P = np.zeros_like(A, dtype=np.float64)
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        ups = self.upsilon
        # Rows and Columns
        row = A.shape[0]
        col = A.shape[1]
        for i in range(row):
            for j in range(col):
               mid = A[i][j] * (x[j] - x[i])
               P[i][j] = alpha * sign_abs(mid, mu) + \
                        beta * sign_abs(mid, ups) 
        return P
    

if __name__ == '__main__':
    # x_init = np.array([350, 100, 200, 250, 400, 500])
    # A = np.array([[0, 5, 0, 6, 0, 0],
    #             [5, 0, 7, 0, 4, 0],
    #             [0, 7, 0, 3, 0, 0],
    #             [6, 0, 3, 0, 2, 0],
    #             [0, 4, 0, 2, 0, 1],
    #             [0, 0, 0, 0, 1, 0]])
    x_init = np.array([-5, -2, 4, 6, 4, 5])
    A = np.array([[0, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0]])
    # ffp = FisrtFixedProtocol(A, 1, 1, 1, 1)
    ffp = FisrtFixedProtocol(A)
    # 仿真参数
    ms = FirstOrderSystem(A, x_init)
    time = 10  # s
    gamma = 1
    time_l = int(10 / ms.dt)
    for i in range(time_l):
        # u = -gamma * np.dot(ms.top.L, ms.x)  # 基本一致性协议
        u = ffp.get_u(ms.x)
        ms.update(u)
    # 画图
    show_result(ms.x_array)
    show_result(ms.u_array)
