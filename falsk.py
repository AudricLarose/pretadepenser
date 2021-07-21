
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import dash  # (version 1.12.0) pip install dash
from flask import Flask,jsonify,make_response,abort
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dash/'
)


server = Flask(__name__)
server.secret_key ='dfdfdfdf'

dforiginal=pd.read_csv('application_test.csv', encoding="UTF-8")
dforiginal2=pd.read_csv('df_P7Clean.csv',encoding="UTF-8")

df=pd.DataFrame.copy(dforiginal.dropna().head(1000))
df2=pd.DataFrame.copy(dforiginal2.dropna().head(1000))
dfPrep=pd.get_dummies(df)
dfPrep2=pd.get_dummies(df2)
train,test=train_test_split(dfPrep)
train2,test2=train_test_split(dfPrep2)
x_test=test
with open("pickle_model.pkl", 'rb') as file:
   pickle_model = pickle.load(file)
# create(server)

df_majority = train2[train2.TARGET == 0]
df_minority = train2[train2.TARGET == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=df_majority.shape[0], random_state=123)
dftrain = pd.concat([df_majority, df_minority_upsampled])
dftrain.TARGET.value_counts()

x_train2 = dftrain.drop('TARGET', axis=1)
y_train2 = dftrain['TARGET']
x_test2 = test2.drop('TARGET', axis=1)
y_test2 = test2['TARGET']
modele = RandomForestClassifier().fit(x_train2, y_train2)

xpl = SmartExplainer()
xpl.compile(
    x=x_test2,
    model=modele)
app = xpl.run_app(host='localhost')

if __name__=="__main__":
    server.secret_key = 'super secret key'
    server.config['SESSION_TYPE'] = 'filesystem'
    server.run(debug=True)
