# Core Pkg
from flask import Flask,jsonify,make_response,abort
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import json


dforiginal=pd.read_csv('application_test.csv', encoding="UTF-8")
# dforiginal2=pd.read_csv('df_P7Clean.csv',encoding="UTF-8")

df=pd.DataFrame.copy(dforiginal.dropna())
dfPrep=pd.get_dummies(df)
train,test=train_test_split(dfPrep)
x_test=test

# Init
app = Flask(__name__)

books=[{
    "id":1,
    "title":"Python for begginer"
}]
with open("pickle_model.pkl", 'rb') as file:
   pickle_model = pickle.load(file)

def predit(x):
    client = dfPrep[dfPrep['SK_ID_CURR'] == x]
    discrimination_threshold = 0.2
    predictions = (pickle_model.predict_proba(client)[::, 1] > discrimination_threshold) * 1
    if int(predictions)==0:
        predictt='Votre profils est accepté'
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
# book = [book for book in jj if book['SK_ID_CURR'] == 100107]
#
# print(book[0]['prediction'])
# print(book)
# # user=x_test[x_test['sku']==data]
#
# # print(x_test.iloc[0:1,1:])
# for cursor in x_test:
#     print(cursor)
#
# # print(len(predictions))
#

@app.route('/')
def index():
	return 'API Tutorials with Flask.eg /api/v1/books to see books'

@app.route('/api/v1/books/<int:id>',methods=['GET'])
def get_books(id):
    book = [book for book in jj if book['SK_ID_CURR'] == id]
    return '<h1>'+ str(book[0]['prediction'] +'</h1>')

@app.errorhandler(404)
def not_found(error):
	return make_response(jsonify({'error':'Not Found'}),404)

if __name__ == '__main__':
	app.run(debug=True)
