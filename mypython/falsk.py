from flask import  Flask,render_template,g,session,request,url_for,abort,redirect
import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from string import punctuation
import io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import json

import io
from shapash.data.data_loader import data_loading
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
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
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask,jsonify,make_response,abort
import json
from dash.dependencies import Input, Output
from mypython import create
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dash/'
)


server = Flask(__name__)
server.secret_key ='dfdfdfdf'

dforiginal=pd.read_csv('application_test.csv', encoding="UTF-8")
# dforiginal2=pd.read_csv('df_P7Clean.csv',encoding="UTF-8")

df=pd.DataFrame.copy(dforiginal.dropna().head(1000))
# df2=pd.DataFrame.copy(dforiginal2.dropna().head(1000))
dfPrep=pd.get_dummies(df)
# dfPrep2=pd.get_dummies(df2)
train,test=train_test_split(dfPrep)
# train2,test2=train_test_split(dfPrep2)
x_test=test
with open("pickle_model.pkl", 'rb') as file:
   pickle_model = pickle.load(file)
# create(server)

# df_majority = train2[train2.TARGET == 0]
# df_minority = train2[train2.TARGET == 1]
# df_minority_upsampled = resample(df_minority, replace=True, n_samples=df_majority.shape[0], random_state=123)
# dftrain = pd.concat([df_majority, df_minority_upsampled])
# dftrain.TARGET.value_counts()

# x_train2 = dftrain.drop('TARGET', axis=1)
# y_train2 = dftrain['TARGET']
# x_test2 = test2.drop('TARGET', axis=1)
# y_test2 = test2['TARGET']
# modele = RandomForestClassifier().fit(x_train2, y_train2)

# xpl = SmartExplainer()
# xpl.compile(
#     x=x_test2,
#     model=modele)
# app = xpl.run_app(host='localhost')


@server.route("/dash")
def my_dash_app():
  return '<a href=' + url_for('index') + '> retour </a>'

# with open("pickle_model.pkl", 'rb') as file:
#    pickle_model = pickle.load(file)

def predit(x):
    client = dfPrep[dfPrep['SK_ID_CURR'] == x]
    discrimination_threshold = 0.2
    predictions = (pickle_model.predict_proba(client)[::, 1] > discrimination_threshold) * 1
    if int(predictions)==0:
        predictt='Votre profil est accepté'
    else :
        predictt='Votre profil est refusé'
    return predictt

df['prediction']=df['SK_ID_CURR'].apply(lambda x : predit(x))
dfenw=df[['SK_ID_CURR','prediction']]
booksdata=dfenw.to_dict(orient="index")
print(df.head())
print(dfenw.head())
print(booksdata)
todictionnaire=[data for data in booksdata.values()]
print(todictionnaire)
json_object = json.dumps(todictionnaire, indent = 1)
jj=json.loads(json_object)


@server.route('/')
def index():
    return render_template('index.html',link=url_for("/dash/"))

@server.before_request
def before_request():
    g.user = None
    if 'user_id' in session:
        user = [x for x in users if x.id == session['user_id']][0]
        g.user = user


@server.route('/resultat', methods=['POST'])
def login():
        session.pop('user_id', None)
        username = request.form['nom']
        book = [book for book in jj if book['SK_ID_CURR'] == int(username)]
        return render_template("resultat.html", resultat=book[0]['prediction'])

# @server.route('/sheeesh')
# def sheesh():
#     app=xpl.run_app(host='localhost')
#     return '<a href=' + url_for('index') + '> retour </a>'


if __name__=="__main__":
    server.secret_key = 'super secret key'
    server.config['SESSION_TYPE'] = 'filesystem'
    server.run(debug=True)
