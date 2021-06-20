import nltk
from nltk.corpus import stopwords
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.decomposition import TruncatedSVD
from gensim.corpora.dictionary import Dictionary
from nltk import *
import matplotlib.pyplot as plt
from string import punctuation
import zipfile
import io
import pandas as pd
from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt
from nltk.stem import PorterStemmer
import io
import pandas as pd
from google.colab import files
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
from sklearn.utils import resample

uploaded0 = files.upload()


"""#Checklist DF1 #

variable target : 

1.   lignes et colonnes : 3340, 47
2.   types de variables : qualitatives : 15, quantitatives : 32
3.    Analyse des valeurs manquantes : Les features des appartements sont moins rempli que les infos de l'apprtement lui meme , les commentaires , outliers et les informations sur le lieu comme le code postal


"""

dforiginal=pd.read_csv('df_P7Clean.csv', encoding="UTF-8")
df=pd.DataFrame.copy(dforiginal.dropna())
print(df.shape)
print(dforiginal.shape)

# df.to_csv('df_P7Clean.csv')
# files.download("df_P7Clean.csv")

df.head()

print("nombre de colonnes  : " + str(len(df.columns)))
print('        ')
df.dtypes.value_counts().plot.pie()
plt.title('Proportion de type colonne')
plt.show()

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(),cbar=False)

from scipy.stats import norm
for col in df.select_dtypes('float64'):
  plt.figure(figsize=(20,10))
  sns.distplot(df[col],fit=norm)

df.sort_values(by='TARGET')

df.TARGET.value_counts().plot.pie()

print(' il y a : ' + str(str(df.shape[0])) + ' Données')
print(' il y a : ' + str(len(df.SK_ID_CURR.unique())) +' ID unique en tout')



"""###Equilibre

Nous allons donner plus de poids a
"""



choice_color = ['r','c','g','b']

dfPrep=pd.get_dummies(df)
del dfPrep['SK_ID_CURR']

for col in dfPrep.select_dtypes('object'):
  print('Il y a ' + str(len(df_sampled[col].unique())) + ' valeurs uniques pour la colonne ' + col )
  print(' ')
  print(' --- ')
  print(' ')

train,test=train_test_split(dfPrep)
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
   return y_test,predictions

y_testRa,Y_predRa=score_best_estimator_('RandomForestClassifier')
y_testDec,Y_predDec=score_best_estimator_('decision')

filename = 'finalized_model.sav'
pickle.dump(modeleDecision, open(filename, 'wb'))
