import os
import re

import sqlite3

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.spatial import distance
from numpy.linalg import inv, pinv

def get_data_cols(df:pd.DataFrame, extra:list, show_cols=False):
    pattern = "ImageNumber|Location|Center|Execution_Time|Parent|Child|Metadata"
    if len(extra) > 0:
        pattern+= "|".join(extra)
    meta_cols = df.columns[df.columns.str.contains(pat=pattern, flags=re.IGNORECASE)]
    data_cols = df.drop(columns=meta_cols).select_dtypes(include="float64").columns.tolist()
    if show_cols:
        print(data_cols)
    return data_cols

def mahalanobis(df:pd.DataFrame, data_cols:list, n_pcas=30):
    df = df[data_cols]
    df = df.loc[:, df.nunique() != 1]
    df.fillna(value=df.mean(), inplace=True)
    pca = PCA(n_components=n_pcas)
    reduced = pd.DataFrame(pca.fit_transform(df))
    # covariance matrix and  inverse of covariance matrix
    cov = np.cov(reduced.values, rowvar=False)
    inv_cov = inv(cov)
    # mean vector
    mean = np.mean(reduced, axis=0)
    dist = df.apply(lambda row: distance.mahalanobis(row, mean, inv_cov), axis=1)
    return dist

def euclidean(df:pd.DataFrame, data_cols:list, n_pcas=30):
    df = df[data_cols]
    df = df.loc[:, df.nunique() != 1]
    df.fillna(value=df.mean(), inplace=True)
    pca = PCA(n_components=n_pcas)
    reduced = pd.DataFrame(pca.fit_transform(df))
    dist = distance.euclidean(reduced.values)
    return dist
    