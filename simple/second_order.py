import numpy as np
import matplotlib.pyplot as plt 
from base.base import SecondOrderSystem
from base.utils import show_result


if __name__ == '__main__':
    x_init = np.array([1, 2, 3, 4])
    v_init = np.array([1, 2, 3, 4])
    A = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]])
    ms = SecondOrderSystem(A, x_init, v_init)

    # 仿真参数
    time = 10  # s
    gamma = 1
    time_l = int(10 / ms.dt)
    for i in range(time_l):
        # 基本一致性协议
        u = -(np.dot(ms.top.L, ms.x) + gamma * np.dot(ms.top.L, ms.v))
        ms.update(u)
    # 画图
    show_result(ms.x_array)
    show_result(ms.v_array)
    show_result(ms.u_array)
