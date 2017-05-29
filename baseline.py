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

def featureManipulation(dtfm):
    '''依次处理某一dataframe内__所有__列的__所有__零值'''
    for col, _ in user.iteritems():
        dtfm_col = dtfm[dtfm[col] > 0]
        # 注！！size方法返回的数量里，包含NaN的统计，而count不包含NaN，所以如果使用size，会导致后面的样本数，大于总体数
        # 而对于重写的__len__方法，其值与count()方法相同，但count返回的是一列值，而len是一个值
        pr_col = dtfm_col[col].value_counts()/len(dtfm_col[col])
        pr_col *= len(dtfm[col][(dtfm[col] == 0)])
        pr_col = pr_col.apply(np.round)
        pr_col = pr_col.to_frame()
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


user = pd.read_csv('./pre/user.csv')
user.head()
featureManipulation(user)
