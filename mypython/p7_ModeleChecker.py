# -*- coding: utf-8 -*-
"""P7

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10cas1tGp_KHBIE2Qu_nXt5aNyJ1sA8Jd
"""
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from string import punctuation
import io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import io
from shapash.data.data_loader import data_loading
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import  learning_curve
from sklearn.model_selection import  validation_curve
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.ensemble import IsolationForest
choice_color = ['r','c','g','b']

dforiginal=pd.read_csv('application_test.csv', encoding="UTF-8")
dforiginal2=pd.read_csv('df_P7Clean.csv', encoding="UTF-8")


df=pd.DataFrame.copy(dforiginal.dropna())
# print(df.shape)
# print(df.iloc[0:1,1:])

dfPrep=pd.get_dummies(df)
train,test=train_test_split(dfPrep,random_state=41)

x_test=test
client=dfPrep[dfPrep['SK_ID_CURR']==105979]

with open("pickle_model.pkl", 'rb') as file:
    pickle_model = pickle.load(file)

discrimination_threshold = 0.2
predictions = (pickle_model.predict_proba(client)[::,1] > discrimination_threshold )*1
print(predictions)
