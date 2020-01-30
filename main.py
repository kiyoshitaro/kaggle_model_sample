import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from model import Logistic, RandomForest, lightgbm_lib
from utils import save_file


def numerize(df):
    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col].unique()) >= 10:
            df.drop(columns=[col], inplace=True)
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df)
    return df

    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         th = set(df[col])
    #         if len(th) > 10:
    #             print("ll")
    #             df.drop(columns=[col], inplace=True)
    #         else:
    #             df[col] = pd.Series([list(th).index(i) for i in df[col]])
    # df.fillna(0, inplace=True)
    # df = pd.get_dummies(df)
    # return df



if __name__ == '__main__':

    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    df = pd.concat([train_df, test_df])
    # df.fillna(0, inplace=True)
    df = numerize(df)
    train_df = df[df['id'] < 30000]
    test_df = df[df['id'] >= 30000]
    cols=["label","id"]
    y = train_df["label"]
    X = train_df.drop(cols,axis=1)
    test_X  = test_df.drop(cols,axis=1)
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    train_X = sc.fit_transform(train_X)
    test_X = sc.transform(test_X)
    val_X = sc.transform(val_X)

    y_pred = Logistic(train_X, train_y, test_X)
    # y_pred = RandomForest(train_X, train_y, test_X)
    # y_pred = lightgbm_lib(train_X, train_y, test_X)
    
    save_file("submission_logistic.csv",test_df["id"],y_pred)



