import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from mypython import app
import pathlib

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../mypython").resolve()

df = pd.read_csv(DATA_PATH.joinpath("df_P7Clean.csv"))
df['Ratio-GP-Annuity'] = ((df['AMT_INCOME_TOTAL']) / (df['AMT_CREDIT']))
df["Monthly"] = df['AMT_ANNUITY'] / 12
df["Mean_AMT_INCOME_TOTAL"] = np.mean(df['AMT_INCOME_TOTAL'])
df["Mean_Monthly"] = np.mean(df['Monthly'])
df["Mean_ANNUITY"] = np.mean(df['AMT_ANNUITY'])
df["Mean_AMT_CREDIT"] = np.mean(df['AMT_CREDIT'])
df["Mean_AMT_GOODS_PRICE"] = np.mean(df['AMT_GOODS_PRICE'])
layout = html.Div([
    html.H1("Vue général des crédits clients", style={'text-align': 'center'}),
    dcc.Dropdown(id='slct_Target12', options=[
        {'label': i, 'value': i} for i in df.select_dtypes('float64')
    ],           className='six columns',
                 multi=False,persistence=True, persistence_type='memory',
                 value='AMT_ANNUITY',
                 style={'width': "40%"}),
    html.Div(id='output_container12', children=[]),
    html.Br(),
    dcc.Graph(id='graphique12', figure={}),
    dcc.Graph(id='graphique77', figure={}),
    dcc.Dropdown(id='slct_Target14', options=[
        {'label': i, 'value': i} for i in df.select_dtypes('object')
    ],
                 multi=False,
                 value='NAME_TYPE_SUITE',
                 style={'width': "40%"}
                 ),
    html.Div(id='output_container14', children=[]),
    html.Br(),
    dcc.Graph(id='graphique14', figure={}),
])


@app.callback(
    [Output(component_id='output_container12', component_property='children'),
     Output(component_id='graphique12', component_property='figure'),
     Output(component_id='graphique77', component_property='figure')],
    [Input(component_id='slct_Target12', component_property='value')]
)
def update_graph2(option_slctd):
    print(option_slctd)
    print(type(option_slctd))
    container = ""
    fig = px.scatter(data_frame=df, x='AMT_INCOME_TOTAL', y=option_slctd, color='TARGET', template='plotly_dark',
                     log_x=True, marginal_y="violin",
                     marginal_x="box", trendline="ols")
    fig2 = px.histogram(data_frame=df, x=option_slctd, color='TARGET', template='plotly_dark',
                        log_x=True, marginal='rug')
    return container, fig, fig2


@app.callback(
    [Output(component_id='output_container14', component_property='children'),
     Output(component_id='graphique14', component_property='figure')],
    [Input(component_id='slct_Target14', component_property='value')]
)
def update_graph3(option_slctd):
    print(option_slctd)
    print(type(option_slctd))
    container = ""
    fig = px.histogram(data_frame=df,x=option_slctd,color='TARGET',template='plotly_dark',)
    return container, fig