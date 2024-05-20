from abc import abstractmethod
import numpy as np


class Protocol:
    """
    仅保留通信拓扑，以更好地抽象出一致性协议的基本结构
    """
    def __init__(self, A):
        # 通信拓扑
        if isinstance(A, np.ndarray):
            self.A = A
        elif A is None:
            self.A = np.array([[0, 1, 0, 1],
                               [1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [1, 0, 1, 0]])
        else:
            raise ValueError("A must be a numpy array or None")
        self.D = np.diag(np.sum(self.A, axis=1))
        self.L = self.D - self.A


class BaseSystem:
    """
    状态、通信拓扑、控制量为所有系统共有的属性
    """
    def __init__(self, A, x_init):
        self.dt = 0.01
        if isinstance(x_init, np.ndarray):
            self.x_init = x_init
        else:
            self.x_init = np.array([1, 2, 3, 4])
        self.pro = Protocol(A)
        # 控制
        self.x = self.x_init
        self.u = np.zeros_like(self.x)
        # 存储，利用list
        self.x_array = list()
        self.u_array = list()
        self.x_array.append(self.x.tolist())

    @abstractmethod
    def update(self):
        pass

    def base_store(self):
        self.x_array.append(self.x.tolist())
        self.u_array.append(self.u.tolist())
        

class FirstOrderSystem(BaseSystem):
    def __init__(self, A, x_init):
        super().__init__(A, x_init)
    
    def update(self, u):
        self.u = u
        self.x = self.x + self.dt * self.u
        # 存储
        self.base_store()


class SecondOrderSystem(BaseSystem):
    def __init__(self, A, x_init, v_init):
        super().__init__(A, x_init)
        if isinstance(v_init, np.ndarray):
            self.v_init = v_init
        else:
            self.v_init = np.array([1, 2, 3, 4])
        self.v = v_init
        self.v_array = list()
        self.v_array.append(self.v.tolist())
    
    def update(self, u):
        self.u = u
        self.x = self.x + self.dt * self.v  # 必须在np下进行
        self.v = self.v + self.dt * self.u
        # 存储
        self.base_store()
        self.v_array.append(self.v.tolist())
        