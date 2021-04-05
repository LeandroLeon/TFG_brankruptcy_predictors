import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from pandas_profiling import ProfileReport
np.random.seed(42)


def data_provider():
    filename = 'DATOS.csv'
    dataframe = shuffle(pd.read_csv(filename, sep=","))
    dataframe.reset_index(inplace=True, drop=True)

    # prof = ProfileReport(dataframe)
    # prof.to_file(output_file='output.html'

    X = dataframe.iloc[:, :-1]
    cor_matrix = X.corr().abs()

    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]
    # print(to_drop)

    dataframe = dataframe.drop(to_drop, axis=1)
    # print(dataframe.head())

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
    X_train = dataframe.iloc[:1500, :-1]
    Y_train = dataframe.iloc[:1500, -1:]
    X_train = scaler.fit_transform(X_train, Y_train)
    X_test = scaler.transform(dataframe.iloc[1500:, :-1])
    Y_test = dataframe.iloc[1500:, -1:]
    # print(X_train, Y_train)

    sm = SMOTE()
    X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train)

    return X_train_res, Y_train_res, X_test, Y_test


def data_provider_test_balanced():
    filename = 'DATOS.csv'
    dataframe = shuffle(pd.read_csv(filename, sep=","))
    dataframe.reset_index(inplace=True, drop=True)

    # prof = ProfileReport(dataframe)
    # prof.to_file(output_file='output.html')

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
    X_train = dataframe.iloc[:1500, :-1]
    Y_train = dataframe.iloc[:1500, -1:]
    X_train = scaler.fit_transform(X_train, Y_train)
    X_test = scaler.transform(dataframe.iloc[1500:, :-1])
    Y_test = dataframe.iloc[1500:, -1:]
    #  print(X_train, Y_train)

    sm = SMOTE()
    X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train)
    X_test_res, Y_test_res = sm.fit_sample(X_test, Y_test)

    return X_train_res, Y_train_res, X_test_res, Y_test_res


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

    return X_scl, Y


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
