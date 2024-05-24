import numpy as np
from base.base import SecondOrderSystem, Topology
from base.utils import show_legend
from second_order import SecondBaseProtocol


"""
二阶基本一致性协议适用情况：
  1. 无向图，连通
  2. 有向图，有生成树（比强连通条件要弱）
  3. 增益gamma满足条件

  待实现:
  1. gamma的最小计算公式
"""
class SwitchSecondOrderSystem(SecondOrderSystem):
    def switch_A(self, A_new):
        self.top = Topology(A_new)


def main(A, x_init, v_init):
    ms = SecondOrderSystem(A, x_init, v_init)
    pro = SecondBaseProtocol(ms.top.L)

    # 仿真参数
    time = 10  # s
    time_l = int(time / ms.dt)
    for i in range(time_l):
        # 基本一致性协议
        u = pro.get_u(ms.x, ms.v)
        ms.update(u)
    # 画图
    show_legend(ms.x_array, 'x')
    show_legend(ms.v_array, 'v')
    show_legend(ms.u_array, 'u')


if __name__ == '__main__':
    x_init = np.array([1, 2, 3, 4])
    v_init = np.array([1, 2, 3, 4])
    A = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]])
    A_1 = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])
    A_array = [A, A_1]
    main(A, x_init, v_init)
    