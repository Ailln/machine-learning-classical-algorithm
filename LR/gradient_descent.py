import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import data_utils


def grad_step(w, b):
    mse = 0
    w_grad = 0
    b_grad = 0
    n = df_data.index.size
    for x, y in zip(df_data["cp"].values, df_data["cp_new"].values):
        mse += (1/n) * (y-(w*x+b))**2
        w_grad += -(2/n) * x * (y-(w*x+b))
        b_grad += -(2/n) * (y-(w*x+b))

    return mse, w_grad, b_grad

def gradient_descent_method(df_data):
    # 初始值
    w = 100
    b = 100
    # 学习率
    lr_w = 5*10e-7
    lr_b = 10e-2
    # 迭代次数
    epoch_num = 300

    for e in range(1, epoch_num+1):
        mse, w_g, b_g = grad_step(w, b)
        w = w - (lr_w * w_g)
        b = b - (lr_b * b_g)
        if not e % 10:
            print(f"epoch: {e}, mse:{mse:.2f}, w: {w:.2f}, b: {b:.2f}")

    print(f"w: {w:.2f}, b: {b:.2f}")

    plt.scatter(df_data["cp"], df_data["cp_new"], s=20, c="green", alpha=0.5)
    line_x = np.linspace(0, 630)
    line_y = np.array(b + w * line_x)
    plt.plot(line_x, line_y.T, color='red')
    plt.text(200, 1080, f"y={b:.2f}+{w:.2f}*x", rotation=30,
             fontsize=14, fontstyle="italic")
    plt.xlabel('cp')
    plt.ylabel('cp_new')
    plt.title('Gradient Descent Result')
    plt.show()

if __name__ == '__main__':
    df_data = data_utils.read_data("pokemon")
    gradient_descent_method(df_data)
