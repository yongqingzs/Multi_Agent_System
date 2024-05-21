import numpy as np
import matplotlib.pyplot as plt 


def show_result(x_array):
    x_array = np.array(x_array)
    for i in range(x_array.shape[1]):
        plt.plot(x_array[:, i])
    plt.show()    


def sign_abs(A, pow):
    return np.sign(A) * np.power(np.abs(A), pow)


if __name__ == '__main__':
    print(sign_abs(-2, 2))
