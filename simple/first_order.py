import numpy as np
import matplotlib.pyplot as plt 
from base.base import FirstOrderSystem
from base.utils import show_result


"""
一阶基本一致性协议适用情况：
  1. 无向图，连通
  2. 有向图，有生成树（比强连通条件要强）

  性质：
  1. 在系统的网络拓扑结构为无向图时，系统最终一致值为所有智能体初始值的平均值
"""
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
        u = -gamma * np.dot(ms.top.L, ms.x)
        ms.update(u)
    # 画图
    show_result(ms.x_array)
    show_result(ms.u_array)
    