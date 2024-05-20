import numpy as np
import matplotlib.pyplot as plt 


def show_result(x_array):
    x_array = np.array(x_array)
    for i in range(x_array.shape[1]):
        plt.plot(x_array[:, i])
    plt.show()    
