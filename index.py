import pandas as pd
import numpy as np
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pickle
from sklearn.model_selection import train_test_split
import json
from app import app
from app import server
import p7_2,p7_1

# PATH = pathlib.Path(__file__).parent
# DATA_PATH = PATH.joinpath("../").resolve()
dforiginal = pd.read_csv("df_P7Clean.csv")
df=pd.DataFrame.copy(dforiginal.dropna().head(100))
df['Ratio-GP-Annuity'] = ((df['AMT_INCOME_TOTAL']) / (df['AMT_CREDIT']))
df["Monthly"] = df['AMT_ANNUITY'] / 12
df["Mean_AMT_INCOME_TOTAL"] = np.mean(df['AMT_INCOME_TOTAL'])
df["Mean_Monthly"] = np.mean(df['Monthly'])
df["Mean_ANNUITY"] = np.mean(df['AMT_ANNUITY'])
df["Mean_AMT_CREDIT"] = np.mean(df['AMT_CREDIT'])
df["Mean_AMT_GOODS_PRICE"] = np.mean(df['AMT_GOODS_PRICE'])
dfPrep=pd.get_dummies(df)
train,test=train_test_split(dfPrep)
x_test=test
with open("pickle_model.pkl", 'rb') as file:
   pickle_model = pickle.load(file)
# create(server)

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




app.layout = html.Div([
    html.P(['Entrer votre numero de clients et appuyez sur " Entrez " ']),
    html.Div([
        dcc.Input(
            id='my_txt_input',
            type='text',
            max=6,
            debounce=True,  # changes to input are sent to Dash server only on enter or losing focus
            pattern=r"^[1-9].*",  # Regex: string must start with letters only
            spellCheck=True,
            inputMode='latin',  # provides a hint to browser on type of data that might be entered by the user.
            name='text',  # the name of the control, which is submitted with the form data
            list='browser',  # identifies a list of pre-defined options to suggest to the user
            n_submit=0,  # number of times the Enter key was pressed while the input had focus
            n_submit_timestamp=-1,  # last time that Enter was pressed
            autoFocus=True,  # the element should be automatically focused after the page loaded
            n_blur=0,  # number of times the input lost focus
            n_blur_timestamp=-1,  # last time the input lost focus.


        ),
    ]),
    html.Br(),
    html.Div(id='reponse', children=[]),
    html.Br(),

    html.Div([
       dcc.Link('La moyenne et Vous |' , href='/p7_1'),
       dcc.Link('Globalité ' , href='/p7_2')
    ],className="row"),
    dcc.Location(id='url',refresh=False),
    html.Div(id='page-content',children=[]),

])
@app.callback(
    Output(component_id='page-content', component_property='children'),
   [Input(component_id='url', component_property='pathname')])
def update_graph2   (pathname):
    if pathname=='/p7_1':
        return p7_1.layout
    if pathname=='/p7_2':
        return p7_2.layout

@app.callback(
        Output(component_id='reponse', component_property='children'),
        [Input(component_id='my_txt_input', component_property='value')]
)
def numberchooser(option_slctd=""):
        container = ""
        if option_slctd:
            if (len(option_slctd) == 6):
                print(option_slctd)
                book = [book for book in jj if book['SK_ID_CURR'] == int(option_slctd)]
                print(book)
                if not book:
                    container = "Nous n'avons pas trouvé votre dossier"
                else:
                    container = book[0]['prediction']
            else:
                container = " "

        return container


if __name__ == '__main__':
    app.run_server(debug=True)
