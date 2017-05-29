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

user = pd.read_csv('./pre/user.csv')
user.head()

user_education = user[user.education>0]
user_education.head()

# 注！！size方法返回的数量里，包含NaN的统计，而count不包含NaN，所以如果使用size，会导致后面的样本数，大于总体数
# 而对于重写的__len__方法，其值与count()方法相同，但count返回的是一列值，而len是一个值
pr_user_education = user_education['education'].value_counts()/len(user_education['education'])
pr_user_education *= len(user.education[(user.education == 0)])
pr_user_education = pr_user_education.apply(np.round)
pr_user_education = pr_user_education.to_frame()
for row in pr_user_education.iterrows():
    zeroSample = user.education[(user.education == 0)]
    replace = row[0]
    num = row[1]['education'].astype(int)
    if num > len(zeroSample):
        print(replace)
        num = len(zeroSample)
    smpl = zeroSample.sample(num)
    smpl = smpl.replace(0, replace)
    user.education.update(smpl)
(user.education == 0).all()
user
