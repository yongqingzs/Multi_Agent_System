import numpy as np
from base.base import FirstOrderSystem, Topology
from base.utils import show_legend
from first_order import FirstBaseProtocol


"""
一阶基本一致性协议适用情况：
  1. 无向图，连通
  2. 有向图，有生成树（比强连通条件要强）

  性质：
  1. 在系统的网络拓扑结构为无向图时，系统最终一致值为所有智能体初始值的平均值
"""
class SwitchFirstOrderSystem(FirstOrderSystem):
    def switch_A(self, A_new):
        self.top = Topology(A_new)


def main(x_init, A_array):
    A = A_array[0]
    A_1 = A_array[1]
    ms = SwitchFirstOrderSystem(A, x_init)
    pro = FirstBaseProtocol(ms.top.L)
    
    # 仿真参数
    time = 10  # s
    time_l = int(time / ms.dt)
    for i in range(time_l):
        if i > time_l/10:
            ms.switch_A(A_1)
            pro = FirstBaseProtocol(ms.top.L)
        # 基本一致性协议
        u = pro.get_u(ms.x)
        ms.update(u)
    # 画图
    show_legend(ms.x_array, 'x')
    show_legend(ms.u_array, 'u')


if __name__ == '__main__':
    x_init = np.array([1, 2, 3, 4])
    A = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]])
    A_1 = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])
    A_array = [A, A_1]
    main(x_init, A_array)
    