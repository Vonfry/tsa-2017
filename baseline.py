#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import pickle
import math

def featureManipulation(dtfm, colList, func):
    '''依次处理某一dataframe内__所有__col的__所有__零值'''
    for col in colList:
        pr_col = func(dtfm, col)
        for row in pr_col.iterrows():
            zeroSample = dtfm[col][(dtfm[col] == 0)]
            replace = row[0]
            num = row[1][col].astype(int)
            if num > len(zeroSample):
                print(replace)
                num = len(zeroSample)
            if num <= 0:
                continue
            smpl = zeroSample.sample(num)
            smpl = smpl.replace(0, replace)
            dtfm[col].update(smpl)
    print(dtfm)


# 这里是对user的例子
user = pd.read_csv('./pre/user.csv')
user.head()
def sln(dtfm, col):
    dtfm_col = dtfm[dtfm[col] > 0]
    pr_col = dtfm_col[col].value_counts()/len(dtfm_col[col])
    pr_col *= len(dtfm[col][(dtfm[col] == 0)])
    pr_col = pr_col.apply(np.round)
    pr_col = pr_col.to_frame()
    return pr_col
featureManipulation(user, ['education'], sln)
