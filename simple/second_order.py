import numpy as np
from base.base import ABCProtocol, SecondOrderSystem
from base.utils import show_legend


"""
二阶基本一致性协议适用情况：
  1. 无向图，连通
  2. 有向图，有生成树（比强连通条件要弱）
  3. 增益gamma满足条件

  待实现:
  1. gamma的最小计算公式
"""
class SecondBaseProtocol(ABCProtocol):
    def __init__(self, L, gamma=1, k=0.01):
        self.L = L
        self.gamma = gamma
        self.k = k
      
    def time_bound_estimate(self):
        pass
    
    def get_u(self, x, v):
        L = self.L
        gamma = self.gamma
        k = self.k
        k = 0
        u = -k * v -(np.dot(L, x) + gamma * np.dot(L, v))
        return u


def main(A, x_init, v_init):
    ms = SecondOrderSystem(A, x_init, v_init)
    pro = SecondBaseProtocol(ms.top.L)

    # 仿真参数
    time = 10  # s
    time_l = int(time / ms.dt)
    for _ in range(time_l):
        # 基本一致性协议
        u = pro.get_u(ms.x, ms.v)
        ms.update(u)
    # 画图
    show_legend(ms.x_array, 'x')
    show_legend(ms.v_array, 'v')
    show_legend(ms.u_array, 'u')


if __name__ == '__main__':
    x_init = np.array([1, 2, 3, 4, 5, 6])
    v_init = np.array([1, 1, 1, 1, 1, 1])
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
    main(A, x_init, v_init)
    