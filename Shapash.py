
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
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
df2['BIRTH_EMPLOTED_INTERVEL'] = df2.DAYS_EMPLOYED - df2.DAYS_BIRTH
df2['BIRTH_REGISTRATION_INTERVEL'] = df2.DAYS_REGISTRATION - df2.DAYS_BIRTH
df2['INCOME_PER_FAMILY_MEMBER'] = df2.AMT_INCOME_TOTAL / df2.CNT_FAM_MEMBERS
df2['SEASON_REMAINING'] = df2.AMT_INCOME_TOTAL/4 -  df2.AMT_ANNUITY
df2['RATIO_INCOME_GOODS'] = df2.AMT_INCOME_TOTAL -  df2.AMT_GOODS_PRICE
df2['CHILDREN_RATIO'] = df2['CNT_CHILDREN'] / df2['CNT_FAM_MEMBERS']
df2['OVER_EXPECT_CREDIT'] = (df2.AMT_CREDIT > df2.AMT_GOODS_PRICE).map({False:0, True:1})
df2['BIRTH_EMPLOTED_INTERVEL'] = df2.DAYS_EMPLOYED - df2.DAYS_BIRTH
df2["Mean_AMT_INCOME_TOTAL"]=np.mean(df2['AMT_INCOME_TOTAL'])
df2['Ratio-GP-Annuity']=((df2['AMT_INCOME_TOTAL'])/(df2['AMT_CREDIT']))
df2["Monthly"]=df2['AMT_ANNUITY']/12
df2["Mean_Monthly"]=np.mean(df2['Monthly'])
df2["Mean_ANNUITY"]=np.mean(df2['AMT_ANNUITY'])
df2["Mean_AMT_CREDIT"]=np.mean(df2['AMT_CREDIT'])
df2["Mean_AMT_GOODS_PRICE"]=np.mean(df2['AMT_GOODS_PRICE'])
df2['TERM'] = df2.AMT_CREDIT / df2.AMT_ANNUITY
dfPrep2=pd.get_dummies(df2)

print(len(dfPrep2.columns))

from sklearn.feature_selection import VarianceThreshold
selector= VarianceThreshold(threshold=0.05)
selector.fit_transform(dfPrep2)
dfPrep2=dfPrep2[dfPrep2.columns[selector.get_support()]]
print(dfPrep2.columns)
train2,test2=train_test_split(dfPrep2)



with open("pickle_model (4).pkl", 'rb') as file:
  pickle_model = pickle.load(file)

df_majority = train2[train2.TARGET == 0]
df_minority = train2[train2.TARGET == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=df_majority.shape[0], random_state=123)
dftrain = pd.concat([df_majority, df_minority_upsampled])
dftrain.TARGET.value_counts()

del dftrain['Unnamed: 0']
del dftrain['SK_ID_CURR']
del test2['SK_ID_CURR']
del test2['Unnamed: 0']

x_train2 = dftrain.drop('TARGET', axis=1)
y_train2 = dftrain['TARGET']
x_test2 = test2.drop('TARGET', axis=1)
y_test2 = test2['TARGET']
modele = RandomForestClassifier().fit(x_train2, y_train2)
dictionnaire={'RATIO_INCOME_GOODS':'Ratio Revenue/Bien','BIRTH_EMPLOTED_INTERVEL':'Nombre jours employés client','SEASON_REMAINING':'Nombre de saisons restantes','TERM':'Durée Credit','BIRTH_REGISTRATION_INTERVEL':'Age client au moment credit en jour','OBS_60_CNT_SOCIAL_CIRCLE':'Score du Cercle Social','HOUR_APPR_PROCESS_START':'Heure_soucription_credit', 'OWN_CAR_AGE':'Age Voiture CLient','AMT_ANNUITY':'Pret Annuel','AMT_INCOME_TOTAL':'Revenue Total','AMT_CREDIT':'Montant Credit','DAYS_EMPLOYED':'Nombres de jours employés','AMT_GOODS_PRICE':"Prix du Bien","DAYS_LAST_PHONE_CHANGE":"Derniere fois que le telephone a changé","DAYS_REGISTRATION":"Jour Inscription","DAYS_BIRTH":"Jours de naissance","DAYS_ID_PUBLISH":"Jour ID Publication","FLAG_OWN_REALTY_Y":"Possede un Bien","FLAG_OWN_REALTY_N":"Ne possede pas de bien","AMT_REQ_CREDIT_BUREAU_YEAR":"Demande de Renseignement Credit bureau","OBS_30_CNT_SOCIAL_CIRCLE":"Cercle Social","NAME_EDUCATION_TYPE_Higher education":"Hautes Educations"}
label={0:"Non",1 : "Oui"}
xpl = SmartExplainer(features_dict=dictionnaire)
xpl.compile(
    x=x_test2,
    model=pickle_model)
app = xpl.run_app(host='localhost')

if __name__=="__main__":
    server.secret_key = 'super secret key'
    server.config['SESSION_TYPE'] = 'filesystem'
    server.run(debug=True)
