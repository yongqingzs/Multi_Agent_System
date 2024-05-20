import numpy as np
import matplotlib.pyplot as plt 
from base import FirstOrderSystem
from utils import show_result


if __name__ == '__main__':
    x_init = np.array([1, 2, 3, 4, 5, 6])
    A = np.array([[0, 1, 0, 1, 0, 1],
                  [1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1],
                  [1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1],
                  [1, 0, 1, 0, 1, 0]])
    ms = FirstOrderSystem(A, x_init)

    # 仿真参数
    time = 10  # s
    gamma = 1
    time_l = int(10 / ms.dt)
    for i in range(time_l):
        # 基本一致性协议
        u = -gamma * np.dot(ms.pro.L, ms.x)
        ms.update(u)
    # 画图
    show_result(ms.x_array)
    show_result(ms.u_array)
    