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
#
# @app.callback(
#     [Output(component_id='output_container12', component_property='children'),
#      Output(component_id='graphique12', component_property='figure'),
#      Output(component_id='graphique77', component_property='figure')],
#     [Input(component_id='slct_Target12', component_property='value')]
# )
# def update_graph2(option_slctd):
#     print(option_slctd)
#     print(type(option_slctd))
#     container = ""
#     fig = px.scatter(data_frame=df, x='AMT_INCOME_TOTAL', y=option_slctd, color='TARGET', template='plotly_dark',
#                      log_x=True, marginal_y="violin",
#                      marginal_x="box", trendline="ols")
#     fig2 = px.histogram(data_frame=df, x=option_slctd, color='TARGET', template='plotly_dark',
#                         log_x=True, marginal='rug')
#     return container, fig, fig2
#
# @app.callback(
#     Output(component_id='reponse', component_property='children'),
#     [Input(component_id='my_txt_input', component_property='value')]
# )
# def numberchooser(option_slctd=""):
#     container=""
#     if option_slctd:
#         if (len(option_slctd)==6):
#          print(option_slctd)
#          book = [book for book in jj if book['SK_ID_CURR'] == int(option_slctd)]
#          print(book)
#          if not book:
#              container = "Nous n'avons pas trouvé votre dossier"
#          else:
#              container=book[0]['prediction']
#         else : container = " "
#
#
#
#     return container
#
#
#
# @app.callback(
#     [Output(component_id='output_container14', component_property='children'),
#      Output(component_id='graphique14', component_property='figure')],
#     [Input(component_id='slct_Target14', component_property='value')]
# )
# def update_graph3(option_slctd):
#     print(option_slctd)
#     print(type(option_slctd))
#     container = ""
#     fig = px.histogram(data_frame=df,x=option_slctd,color='TARGET',template='plotly_dark',)
#     return container, fig
#
# @app.callback(
#     [Output(component_id='output_container2', component_property='children'),
#      Output(component_id='graphique4', component_property='figure'),
#      Output(component_id='graphique5', component_property='figure'),
#      Output(component_id='graphique7', component_property='figure')],
#     [Input(component_id='slct_Target2', component_property='value')]
# )
# def update_graph2(option_slctd):
#     print(option_slctd)
#     print(type(option_slctd))
#     option_slctd = int(option_slctd)
#     container = ""
#     langs = ['Vous', 'Moyenne Revenues Totales']
#     students = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL']),
#                 int(df[df['SK_ID_CURR'] == option_slctd]['Mean_AMT_INCOME_TOTAL'])]
#     print(df.select_dtypes('float64').columns)
#     fig1 = px.histogram(data_frame=df, x=df['AMT_INCOME_TOTAL'],log_x=True)
#     fig1.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL']),line_dash="dash", line_color="red",
#
#                    fillcolor="green")
#     fig1.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL'],y=0,showarrow=False,text='<b>Vous</b>',
#                             textangle=0,arrowcolor='red')
#     fig2 = px.histogram(data_frame=df, x=df['AMT_CREDIT'], template='plotly_dark',log_x=True)
#     fig2.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['AMT_CREDIT']),line_dash="dash", line_color="red")
#     fig2.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['AMT_CREDIT'],y=0,showarrow=False,text='<b>Vous</b>',
#                             textangle=0,arrowcolor='red')
#
#     fig4 = px.histogram(data_frame=df, x=df['AMT_ANNUITY'],log_x=True)
#     fig4.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['AMT_ANNUITY']),line_dash="dash", line_color="red")
#     fig4.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['AMT_ANNUITY'],y=0,showarrow=False,text='<b>Vous</b>',
#                             textangle=0,arrowcolor='red')
#
#     return container,fig1, fig2, fig4


if __name__ == '__main__':
    app.run_server(debug=True)
