import numpy as np

class Protocol:
    """
    仅保留通信拓扑，以更好地抽象出一致性协议的基本结构
    """
    def __init__(self, x_init, A):
        self.dt = 0.01
        if isinstance(x_init, np.ndarray):
            self.x_init = x_init
        else:
            self.x_init = np.array([1, 2, 3, 4])
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
        # 控制
        self.x = self.x_init
        self.v = np.zeros_like(self.x)
        self.u = np.zeros_like(self.x)
        # 存储，利用list
        self.x_array = list()
        self.u_array = list()
        self.v_array = list()
        self.x_array.append(self.x.tolist())

class FirstOrderSystem:
    def __init__(self, x_init, A):
        self.sys = Protocol(x_init, A)
    
    def update(self, u):
        self.sys.u = u
        self.sys.x = self.sys.x + self.sys.dt * self.sys.u
        # 存储
        self.sys.x_array.append(self.sys.x.tolist())
        self.sys.u_array.append(self.sys.u.tolist())

class SecondOrderSystem:
    def __init__(self, x_init, A):
        self.sys = Protocol(x_init, A)
    
    def update(self, u):
        self.sys.u = u
        self.sys.x = self.sys.x + self.sys.dt * self.sys.v  # 必须在np下进行
        self.sys.v = self.sys.v + self.sys.dt * self.sys.u
        # 存储
        self.sys.x_array.append(self.sys.x.tolist())
        self.sys.u_array.append(self.sys.u.tolist())
