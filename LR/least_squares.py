import numpy as np
from matplotlib import pyplot as plt

from utils import data_utils


def least_squares_method(df_data):
    X = np.matrix([np.ones(df_data.index.size), df_data["cp"].values]).T
    y = np.matrix(df_data["cp_new"].values).T
    mat_result = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    b = round(mat_result.tolist()[0][0], 2)
    w = round(mat_result.tolist()[1][0], 2)
    print(f"w: {w}, b: {b}")

    plt.scatter(df_data["cp"], df_data["cp_new"], s=20, c="green", alpha=0.5)
    line_x = np.linspace(0, 630)
    line_y = np.array(mat_result[0] + mat_result[1] * line_x)
    plt.plot(line_x, line_y.T, color='red')
    plt.text(200, 1080, f"y={b}+{w}*x", rotation=30,
             fontsize=14, fontstyle="italic")
    plt.xlabel('cp')
    plt.ylabel('cp_new')
    plt.title('Least Squares Result')
    plt.show()


if __name__ == '__main__':
    df_data = data_utils.read_data("pokemon")
    least_squares_method(df_data)
