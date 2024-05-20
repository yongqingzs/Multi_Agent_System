import numpy as np
import matplotlib.pyplot as plt 
from base import SecondOrderSystem

if __name__ == '__main__':
    x_init = np.array([1, 2, 3, 4])
    A = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]])
    ms = SecondOrderSystem(x_init, A)

    # 仿真参数
    time = 10  # s
    gamma = 1
    time_l = int(10 / ms.sys.dt)
    for i in range(time_l):
        # 基本一致性协议
        u = -(np.dot(ms.sys.L, ms.sys.x) + gamma * np.dot(ms.sys.L, ms.sys.v))
        ms.update(u)
    # 将so.sys.x_array的各列画出
    x_array = np.array(ms.sys.x_array)
    for i in range(x_init.shape[0]):
        plt.plot(x_array[:, i])
    plt.show()
