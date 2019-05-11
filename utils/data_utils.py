import numpy as np
import pandas as pd


def read_data(name=None):
    if name == "pokemon":
        df_data = pd.read_csv("./dataset/pokemon.csv")

        # 筛选出 cp(进化前的战斗力) 和 cp_new（进化后的战斗力）两列特征数据
        df_data = df_data[["cp", "cp_new"]]

        # 先看一下前5条数据，确认是否有问题
        print("top 5 data:")
        print(df_data.head())

        return df_data
    elif name == "iris":
        df_data = pd.read_csv("./dataset/iris.csv")

        # 先看一下前5条数据，确认是否有问题
        print("top 5 data:")
        print(df_data.head())

        return df_data 
    else:
        raise ValueError("only pokemon and iris are allowed.")
