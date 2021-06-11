import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


np.random.seed(42)


def get_data():
    filename = 'DATOS.csv'
    dataframe = shuffle(pd.read_csv(filename, sep=","))
    dataframe.reset_index(inplace=True, drop=True)

    X = dataframe.iloc[:, :-1]
    cor_matrix = X.corr().abs()

    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]

    dataframe = dataframe.drop(to_drop, axis=1)
    print(dataframe.head())

    # X = dataframe.iloc[:, :-1]
    # Q1 = dataframe.quantile(0.25)
    # Q3 = dataframe.quantile(0.75)
    # IQR = Q3 - Q1
    # print(IQR)
    #
    # print((dataframe < (Q1 - 3 * IQR))|(dataframe > (Q3 + 3 * IQR)))
    #
    # dataframe = dataframe[~((dataframe.iloc[:,:-1] < (Q1 - 3 * IQR)) |(dataframe.iloc[:,:-1] > (Q3 + 3 * IQR))).any(axis=1)]
    # print(dataframe.shape)
    # print(dataframe)

    scaler = MinMaxScaler()
    X = dataframe.iloc[:, :-1]
    Y = dataframe.iloc[:, -1:]
    X_scl = scaler.fit_transform(X, Y)

    return train_test_split(X_scl, Y, test_size=0.15, shuffle=True, stratify=Y)


def get_dataframe():
    filename = 'DATOS.csv'
    dataframe = shuffle(pd.read_csv(filename, sep=","))
    dataframe.reset_index(inplace=True, drop=True)

    X = dataframe.iloc[:, :-1]
    cor_matrix = X.corr().abs()

    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]

    dataframe = dataframe.drop(to_drop, axis=1)
    print(dataframe.head())

    scaler = MinMaxScaler()
    X = dataframe.iloc[:, :-1]
    Y = dataframe.iloc[:, -1:]
    X_scl = scaler.fit_transform(X, Y)

    headers = get_headers()

    df_x = pd.DataFrame(X_scl, columns=headers)
    data = pd.concat([df_x, Y], axis=1)

    return data


def get_headers():
    filename = 'DATOS.csv'
    dataframe = shuffle(pd.read_csv(filename, sep=","))
    dataframe.reset_index(inplace=True, drop=True)


    X = dataframe.iloc[:, :-1]
    cor_matrix = X.corr().abs()

    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]

    X = X.drop(to_drop, axis=1)
    return X.columns.tolist()
