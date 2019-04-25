from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(file_path):
    df_iris = pd.read_csv(file_path)
    data_list = df_iris.values
    inputs = data_list[:, :4]
    labels = data_list[:, -1]
    return inputs, labels


def knn(test_data, input_data, label_data, k):
    dis_class_list = []
    for data, label in zip(input_data, label_data):
        dis = np.sqrt(np.sum(np.square(data - test_data)))
        dis_class_list.append([dis, label])
    sort_dis_class_list = sorted(dis_class_list, key=lambda x: x[0])
    k_class_dict = defaultdict(int)
    for dis_class in sort_dis_class_list[:k]:
        class_name = dis_class[1]
        if class_name in k_class_dict.keys():
            k_class_dict[class_name] += 1
        else:
            k_class_dict[class_name] = 1
    max_num = 0
    pred = ""

    for key, value in k_class_dict.items():
        if value >= max_num:
            max_num = value
            pred = key

    return pred


def run():
    data_path = "./dataset/iris.csv"
    origin_data, origin_label = read_data(data_path)
    train_data, test_data, train_label, test_label = train_test_split(
        origin_data, origin_label, random_state=10)

    k_list = [1, 2, 3, 4, 5]

    for k in k_list:
        i = 0
        for test_item, label_item in zip(test_data, test_label):
            pred = knn(test_item, train_data, train_label, k)
            if pred != label_item:
                i += 1

        acc = round(1-(i/len(test_label)), 2)
        print(f"k: {k}, acc: {acc}")


if __name__ == '__main__':
    run()
