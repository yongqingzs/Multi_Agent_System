import numpy as np
from base.base import ABCProtocol, FirstOrderSystem
from base.utils import clear_directory, show_legend


"""
一阶基本一致性协议适用情况：
  1. 无向图，连通
  2. 有向图，有生成树（比强连通条件要强）

  性质：
  1. 在系统的网络拓扑结构为无向图时，系统最终一致值为所有智能体初始值的平均值
"""
class FirstBaseProtocol(ABCProtocol):
    def __init__(self, L, gamma=1):
        self.L = L
        self.gamma = gamma

    def time_bound_estimate(self):
        pass

    def get_u(self, x):
        L = self.L
        gamma = self.gamma
        u = -gamma * np.dot(L, x)
        return u 


def main(x_init, A):
    ms = FirstOrderSystem(A, x_init)
    pro = FirstBaseProtocol(ms.top.L)

    # 仿真参数
    time = 10  # s
    time_l = int(time / ms.dt)
    for i in range(time_l):
        # 基本一致性协议
        u = pro.get_u(ms.x)
        ms.update(u)
    clear_directory('./images')
    # 画图
    show_legend(ms.x_array, 'x')
    show_legend(ms.u_array, 'u')


if __name__ == '__main__':
    x_init = np.array([1, 2, 3, 4, 5, 6])
    A_type = 4
    if A_type == 0:
        A = np.array([[0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0]])  # 无向图，连通
    elif A_type == 1:
        A = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0]])  # 无向图，不连通，孤立节点1
    elif A_type == 2:
        A = np.array([[0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0]])  # 有向图，强连通，平衡
    elif A_type == 3:
        A = np.array([[0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0]])  # 有向图，强连通，不平衡
    elif A_type == 4:
        A = np.array([[0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0]])  # 有向图，不强连通
    main(x_init, A)
    