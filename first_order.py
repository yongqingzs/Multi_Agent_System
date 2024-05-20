import numpy as np
import matplotlib.pyplot as plt 
from base import FirstOrderSystem

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
    # 将x_array的各列画出
    x_array = np.array(ms.x_array)
    for i in range(x_init.shape[0]):
        plt.plot(x_array[:, i])
    plt.show()
