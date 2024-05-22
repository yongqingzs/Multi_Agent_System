import numpy as np
from base.base import ABCProtocol, FirstOrderSystem, Topology
from base.utils import show_result, sign_abs

"""
"Fixed-time Consensus Tracking of Multi-agent Systems under a Directed Communication Topology"

NOTE:
1. 不单独将B拎出来，不创建一个新的拓扑子类
"""
class FixedLeaderProtocol(ABCProtocol):
    def __init__(self, A, alpha=3, mu=1.3, ups=0.4):
        self.A = A
        # 一致性协议自定义的参数
        self.alpha = alpha
        self.mu = mu
        self.ups = ups
        # B只有单向到对面
        self.B = A[0:, 0]

    def time_bound_estimate(self):
        pass
    
    def get_u(self, x):
        A = self.A
        B = self.B
        alpha = self.alpha
        mu = self.mu
        ups = self.ups
        u = np.zeros_like(x, dtype=np.float64)
        u_temp = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                u_temp += A[i][j] * (x[j] - x[i])
            u_temp -= B[i] * (x[i] - x[0])
            u[i] = alpha * (sign_abs(u_temp, mu) + sign_abs(u_temp, ups))
            u_temp = 0
        return u


class FixedLeaderWithDistProtocol(ABCProtocol):
    """
    暂时没有复现成功。
    """
    def __init__(self, A, alpha=3, mu=1.3, ups=0.4, a=1.2, b=1.3, 
                 alpha_1=0.8, alpha_2=1.4, k=7):
        self.A = A
        # 一致性协议自定义的参数
        self.alpha = alpha
        self.mu = mu
        self.ups = ups
        # 滑膜面
        self.a = a
        self.b = b
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.k = k
        # B只有单向到对面
        self.B = A[0:, 0]

    def time_bound_estimate(self):
        pass
    
    def get_u(self, x, dt):
        A = self.A
        B = self.B
        alpha = self.alpha
        mu = self.mu
        ups = self.ups
        a = self.a
        b = self.b
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        k = self.k
        u = np.zeros_like(x, dtype=np.float64)
        u_eqi = 0
        u_di = 0
        z = np.zeros_like(x, dtype=np.float64)
        s = np.zeros_like(x, dtype=np.float64)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                u_eqi += A[i][j] * (x[j] - x[i])
            # u_eqi
            u_eqi -= B[i] * (x[i] - x[0])
            u_eqi = alpha * (sign_abs(u_eqi, mu) + sign_abs(u_eqi, ups))
            # u_di
            z[i] -= u_eqi * dt
            s[i] = x[i] + z[i]
            u_di = -(a + k) * sign_abs(s[i], alpha_1) - \
                    (b + k) * sign_abs(s[i], alpha_2)
            # u
            # u[i] = u_eqi + u_di
            u[i] = u_eqi
            u_eqi = 0
            u_di = 0
        return u


if __name__ == '__main__':
    x_init = np.array([100, 250, -200, 140, -50, 300])
    A = np.array([[0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 2, 0, 0, 2, 0],
                [0, 0, 3, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 2, 0]])
    pro = FixedLeaderProtocol(A)
    
    ms = FirstOrderSystem(A, x_init)  # 有扰动
    time = 10  # s
    gamma = 1
    time_l = int(10 / ms.dt)
    for i in range(time_l):
        # u = -gamma * np.dot(ms.top.L, ms.x)  # 基本一致性协议
        u = pro.get_u(ms.x)
        ms.update(u)
    # 画图
    show_result(ms.x_array)
    show_result(ms.u_array)
