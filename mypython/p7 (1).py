# -*- coding: utf-8 -*-

from shapash.explainer.smart_explainer import SmartExplainer
import numpy as np
import pandas as pd


dforiginal=pd.read_csv('df_P7Clean.csv',encoding="UTF-8")
df=pd.DataFrame.copy(dforiginal.dropna()) 
print(df.shape)
print(dforiginal.shape)

df.sort_values(by='TARGET')

df.TARGET.value_counts().plot.pie()

print(' il y a : ' + str(str(df.shape[0])) + ' Données')
print(' il y a : ' + str(len(df.SK_ID_CURR.unique())) +' ID unique en tout')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.model_selection import  GridSearchCV
choice_color = ['r','c','g','b']

dfPrep=pd.get_dummies(df)
del dfPrep['SK_ID_CURR']

train,test=train_test_split(dfPrep)

from sklearn.utils import resample
df_majority= train[train.TARGET==0]
df_minority= train[train.TARGET==1]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=df_majority.shape[0],random_state=123)
dftrain=pd.concat([df_majority,df_minority_upsampled])
dftrain.TARGET.value_counts()

x_train=dftrain.drop('TARGET',axis=1)
y_train=dftrain['TARGET']
x_test=test.drop('TARGET',axis=1)
y_test=test['TARGET']

def score_best_estimator_(model):
   if  model == 'RandomForestClassifier' : 
    steps = [('scaler', StandardScaler()),
         ('paramz', RandomForestClassifier())]
    pipeline=Pipeline(steps)
    parameters = {'paramz__n_estimators' :np.arange(19,21),
                 'paramz__n_jobs':np.arange(19,21)}
   if  model == 'sgd' :
    steps = [('scaler', StandardScaler()),
         ('paramz', SGDClassifier())]
    pipeline=Pipeline(steps)
    parameters = {
                 'paramz__loss':['epsilon_insensitive']}
   if (model == 'decision') :
    steps = [('scaler', StandardScaler()),
         ('paramz', DecisionTreeClassifier())]
    pipeline=Pipeline(steps)
    parameters = {'paramz__max_depth' :np.arange(1,30)}
   if  model == 'ridge' :
    steps = [('scaler', StandardScaler()),
         ('paramz',RidgeClassifier())]
    pipeline=Pipeline(steps)
    parameters = {'paramz__alpha' :np.arange(1,30),
                 'paramz__fit_intercept':[True,False],
                 'paramz__copy_X' :[True,False]}            
   grid=GridSearchCV(pipeline,parameters,cv=5,scoring="accuracy")
   grid.fit(x_train,y_train)
   Y_pred=grid.predict(x_test)
   Y_predprob=grid.predict_proba(x_test)
   for i in range (10):
     discrimination_threshold = i/10
     predictions = (Y_predprob[::,1] > discrimination_threshold )*1 
     report = classification_report(y_test, predictions)
     matrix = confusion_matrix(y_test, predictions)
     tn, fp, fn, tp =matrix.ravel()
     score_for_loan=(tn+tp)/(tn+fp+(fn*fn)+tp)
     print('Threshold : ' + str(discrimination_threshold))
     print('Report : ' + str(report))
     print('Matrice de confusion : ' + str(matrix))
     print('Score accuracy : ' + str(grid.score(x_test,y_test)))
     print("Score d'emprunt : " + str(score_for_loan))
     print(" ")
     print("--------------------------")
     print(" ")
   print(grid.best_params_)
   return grid,y_test,predictions

print(df.columns)

df.head()

for col in df.select_dtypes('object'):
  print(col)

modele=RandomForestClassifier().fit(x_train,y_train)

xpl = SmartExplainer()
xpl.compile(
    x=x_test,
    model=modele)
app = xpl.run_app(host='localhost')
