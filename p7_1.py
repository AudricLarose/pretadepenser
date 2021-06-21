import pathlib

import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px  # (version 4.7.0)
from dash.dependencies import Input, Output
from app import app
from app import server

# PATH = pathlib.Path(__file__).parent
# DATA_PATH = PATH.joinpath("../mypython").resolve()

df = pd.read_csv("df_P7Clean.csv")  # GregorySmith Kaggle

df['Ratio-GP-Annuity'] = ((df['AMT_INCOME_TOTAL']) / (df['AMT_CREDIT']))
df["Monthly"] = df['AMT_ANNUITY'] / 12
df["Mean_AMT_INCOME_TOTAL"] = np.mean(df['AMT_INCOME_TOTAL'])
df["Mean_Monthly"] = np.mean(df['Monthly'])
df["Mean_ANNUITY"] = np.mean(df['AMT_ANNUITY'])
df["Mean_AMT_CREDIT"] = np.mean(df['AMT_CREDIT'])
df["Mean_AMT_GOODS_PRICE"] = np.mean(df['AMT_GOODS_PRICE'])

layout = html.Div([

    html.H1("Vue d'un client en particulier", style={'text-align': 'center'}),

    dcc.Dropdown(id='slct_Target2', options=[
        {'label': i, 'value': i} for i in df['SK_ID_CURR']
    ], className='six columns',
                 multi=False, persistence=True, persistence_type='memory',
                 value='100083',
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container2', children=[]),
    html.Br(),
    dcc.Graph(id='graphique4', figure={}),
    dcc.Graph(id='graphique5', figure={}),
    dcc.Graph(id='graphique7', figure={}),

])


@app.callback(
    [Output(component_id='output_container2', component_property='children'),
     Output(component_id='graphique4', component_property='figure'),
     Output(component_id='graphique5', component_property='figure'),
     Output(component_id='graphique7', component_property='figure')],
    [Input(component_id='slct_Target2', component_property='value')]
)
def update_graph2(option_slctd):
    print(option_slctd)
    print(type(option_slctd))
    option_slctd = int(option_slctd)
    container = ""
    langs = ['Vous', 'Moyenne Revenues Totales']
    students = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL']),
                int(df[df['SK_ID_CURR'] == option_slctd]['Mean_AMT_INCOME_TOTAL'])]
    print(df.select_dtypes('float64').columns)
    fig1 = px.histogram(data_frame=df, x=df['AMT_INCOME_TOTAL'],log_x=True)
    fig1.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL']),line_dash="dash", line_color="red",

                   fillcolor="green")
    fig1.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL'],y=0,showarrow=False,text='<b>Vous</b>',
                            textangle=0,arrowcolor='red')
    fig2 = px.histogram(data_frame=df, x=df['AMT_CREDIT'], template='plotly_dark',log_x=True)
    fig2.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['AMT_CREDIT']),line_dash="dash", line_color="red")
    fig2.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['AMT_CREDIT'],y=0,showarrow=False,text='<b>Vous</b>',
                            textangle=0,arrowcolor='red')

    fig4 = px.histogram(data_frame=df, x=df['AMT_ANNUITY'],log_x=True)
    fig4.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['AMT_ANNUITY']),line_dash="dash", line_color="red")
    fig4.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['AMT_ANNUITY'],y=0,showarrow=False,text='<b>Vous</b>',
                            textangle=0,arrowcolor='red')

    return container,fig1, fig2, fig4
