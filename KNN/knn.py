from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import data_utils


def knn(test_data, df_train, k):
    dis_class_list = []
    for _, train_item in df_train.iterrows():
        data = train_item.values[:4]
        label = train_item["species"]
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

    pred = max(k_class_dict, key=k_class_dict.get)
    return pred


def run():
    df_data = data_utils.read_data("iris")
    df_train, df_test = train_test_split(df_data)

    k_list = [i+1 for i in range(5)]

    for k in k_list:
        i = 0
        for _, test_item in df_test.iterrows():
            test_data = test_item.values[:4]
            test_label = test_item["species"]
            pred = knn(test_data, df_train, k)
            if pred == test_label:
                i += 1

        acc = round(i/len(df_test), 2)
        print(f"k: {k}, accuracy: {acc}")


if __name__ == '__main__':
    run()
