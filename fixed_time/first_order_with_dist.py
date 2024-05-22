import numpy as np
from base.base import ABCProtocol, FirstOrderSystem
from base.utils import show_result, sign_abs

"""
"Distributed robust finite time nonlinear consensus protocols for multi agent systems"
"""
class FixedWithDisturbanceProtocol(ABCProtocol):
    def __init__(self, A, alpha=2, beta=2, p=3, q=5, m=9, n=7):
        self.A = A
        # 一致性协议自定义的参数
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.m = m
        self.n = n
    
    def time_bound_estimate(self):
        pass
    
    def get_u(self, x, type=1):
        A = self.A
        alpha = self.alpha
        beta = self.beta
        p = self.p
        q = self.q
        m = self.m
        n = self.n
        gamma = 1
        u = np.zeros_like(x, dtype=np.float64)
        u_temp = 0
        if type == 1:
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    u_temp += A[i][j] * (x[j] - x[i])
                u[i] = alpha * sign_abs(u_temp, m/n) + \
                        beta * sign_abs(u_temp, p/q) + \
                        gamma * np.sign(u_temp)
                u_temp = 0
        elif type == 2:
            u_temp_1 = 0
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    u_temp_1 += (x[j] - x[i])
                    u_temp += alpha * A[i][j] * sign_abs(x[j]-x[i], m/n) + \
                            beta * A[i][j] * sign_abs(x[j]-x[i], p/q)
                u[i] = u_temp + gamma * np.sign(u_temp_1)
                u_temp = 0
        else:
            raise ValueError("type must be 1 or 2")
        return u
    

class FirstWithDisturbanceSystem(FirstOrderSystem):
    def __init__(self, A, x_init, dist_type=1, dist_ratio=0.1):
        super(FirstWithDisturbanceSystem, self).__init__(A, x_init)
        if dist_type == 1:
            self.disturbance = np.sin
        elif dist_type == 2:
            self.disturbance = np.cos
        else:
            raise ValueError("dist_type must be 1 or 2")
        self.dist_ratio = dist_ratio
        self.time = 0

    def update(self, u):
        self.u = u
        self.x = self.x + self.dt * (self.u + 0.1 * self.disturbance(self.dist_ratio * self.time))
        self.time += self.dt
        # 存储
        self.base_store()


if __name__ == '__main__':
    x_init = np.array([-5, -3, 3, 8, 4, 5])
    A = np.array([[0, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0]])
    A = 2 * A
    ffp = FixedWithDisturbanceProtocol(A)
    
    # ms = FirstOrderSystem(A, x_init)  # 无扰动
    ms = FirstWithDisturbanceSystem(A, x_init)  # 有扰动
    time = 10  # s
    gamma = 2
    time_l = int(10 / ms.dt)
    for i in range(time_l):
        u = -gamma * np.dot(ms.top.L, ms.x)  # 基本一致性协议
        # u = ffp.get_u(ms.x)
        ms.update(u)
    # 画图
    show_result(ms.x_array)
    show_result(ms.u_array)
