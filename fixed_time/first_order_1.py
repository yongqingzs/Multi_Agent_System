import numpy as np
from base.base import ABCProtocol, FirstOrderSystem
from base.utils import show_result, sign_abs

"""
在不同论文中，会出现参数符号不一致的情况

已完成：
1，控制量计算

未完成：
1. 时间上限估计
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
                # if (i != j) and (x[j] != x[i]):
                    # u_temp += alpha * A[i][j] * np.power((x[j]-x[i]), 2-p/q) + \
                            # beta * A[i][j] * np.power((x[j]-x[i]), p/q)
                # u_temp += A[i][j] * (x[j]-x[i])  # only for test
            u[i] = u_temp
            u_temp = 0
        return u
    

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
