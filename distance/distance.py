import os
import re

import sqlite3

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.spatial import distance
from numpy.linalg import inv, pinv

def get_data_cols(df:pd.DataFrame, extra=[], show_cols=False):
    pattern = "ImageNumber|Location|Center|Execution_Time|Parent|Child|Metadata"
    if len(extra) > 0:
        pattern+="|"+ "|".join(extra)
    print(pattern)
    meta_cols = df.columns[df.columns.str.contains(pat=pattern, flags=re.IGNORECASE)]
    data_cols = df.drop(columns=meta_cols).select_dtypes(include="float64").columns.tolist()
    if show_cols:
        print(data_cols)
    return data_cols

def mahalanobis(df:pd.DataFrame, data_cols:list, n_pcas=30):
    df = df[data_cols]
    df = df.fillna(value=df.mean())
    pca = PCA(n_components=n_pcas)
    pca.fit(df)
    transformed_data = pca.transform(df)
    mean_vec = np.mean(transformed_data, axis=0)
    cov_mat = np.cov(transformed_data.T)

    mahalanobis_distances = []
    for i in range(len(transformed_data)):
        mahalanobis_distance = distance.mahalanobis(transformed_data[i], mean_vec, np.linalg.inv(cov_mat))
        mahalanobis_distances.append(mahalanobis_distance)
    return mahalanobis_distances

def euclidean(df:pd.DataFrame, data_cols:list, n_pcas=30):
    df = df[data_cols]
    df = df.loc[:, df.nunique() != 1]
    df.fillna(value=df.mean(), inplace=True)
    pca = PCA(n_components=n_pcas)
    reduced = pd.DataFrame(pca.fit_transform(df))
    dist = distance.euclidean(reduced.values, np.zeros(len(reduced)))
    return dist
    