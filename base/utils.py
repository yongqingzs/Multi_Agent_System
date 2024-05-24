import glob
import os
from matplotlib.transforms import Bbox
import numpy as np
import matplotlib.pyplot as plt 


def show_result(x_array):
    x_array = np.array(x_array)
    for i in range(x_array.shape[1]):
        plt.plot(x_array[:, i])
    plt.show()    


def sign_abs(A, pow):
    return np.sign(A) * np.power(np.abs(A), pow)


def show_legend(ar, name: str):
    # 设置全局字体样式
    plt.rcParams['font.family'] = 'Times New Roman'  # 字体类型
    plt.rcParams['font.size'] = 15.5  # 字体大小
    ar = np.array(ar)
    plt.figure()
    for i in range(ar.shape[1]):
        plt.plot(ar[:, i], label=f'{name}_{i}')
    # 设置坐标轴名称
    plt.xlabel('step')
    plt.ylabel('value')
    plt.legend(loc='upper right')
    # plt.show()
    # 保存图片到./images下
    plt.savefig(f'./images/{name}.png', bbox_inches='tight')


def clear_directory(directory):
    # 获取目录下的所有文件
    files = glob.glob(directory + '/*')
    for f in files:
        # 删除每一个文件
        os.remove(f)


if __name__ == '__main__':
    print(sign_abs(-2, 2))
