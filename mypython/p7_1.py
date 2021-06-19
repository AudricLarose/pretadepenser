import pathlib

import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px  # (version 4.7.0)
from dash.dependencies import Input, Output

from mypython import app

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../mypython").resolve()

df = pd.read_csv(DATA_PATH.joinpath("df_P7Clean.csv"))  # GregorySmith Kaggle

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
    dcc.Graph(id='graphique2', figure={}),
    dcc.Graph(id='graphique3', figure={}),
    dcc.Graph(id='graphique4', figure={}),
    dcc.Graph(id='graphique5', figure={}),
    dcc.Graph(id='graphique7', figure={}),

])


@app.callback(
    [Output(component_id='output_container2', component_property='children'),
     Output(component_id='graphique2', component_property='figure'),
     Output(component_id='graphique3', component_property='figure'),
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
    dff = df.copy()
    langs = ['Vous', 'Moyenne Revenues Totales']
    students = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL']),
                int(df[df['SK_ID_CURR'] == option_slctd]['Mean_AMT_INCOME_TOTAL'])]
    fig1 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark', color_discrete_sequence=['blue'])
    fig2 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark', color_discrete_sequence=['cyan'])
    fig3 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark', color_discrete_sequence=['royalblue'])
    fig4 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark', color_discrete_sequence=['darkblue'])
    fig = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark', color_discrete_sequence=['azure'])
    return container, fig, fig1, fig2, fig3, fig4
