import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask import Flask,jsonify,make_response,abort
import json

app = dash.Dash(__name__)
df = pd.read_csv("df_P7Clean.csv")
# print(df.head())
# px.histogram(df,x='AMT_INCOME_TOTAL',color='TARGET')
# plt.show()
df['Ratio-GP-Annuity'] = ((df['AMT_INCOME_TOTAL']) / (df['AMT_CREDIT']))
df["Monthly"] = df['AMT_ANNUITY'] / 12
df["Mean_AMT_INCOME_TOTAL"] = np.mean(df['AMT_INCOME_TOTAL'])
df["Mean_Monthly"] = np.mean(df['Monthly'])
df["Mean_ANNUITY"] = np.mean(df['AMT_ANNUITY'])
df["Mean_AMT_CREDIT"] = np.mean(df['AMT_CREDIT'])
df["Mean_AMT_GOODS_PRICE"] = np.mean(df['AMT_GOODS_PRICE'])

app.layout = html.Div([

    html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),

    # dcc.Dropdown(id="slct_Target",
    #              options=[
    #                  {"label": "Good", "value": 0},
    #                  {"label": "Bad", "value": 1}],
    #              multi=False,
    #              value=0,
    #              style={'width': "40%"}
    #              ),
    #
    # html.Div(id='output_container', children=[]),
    # html.Br(),
    # dcc.Graph(id='graphique', figure={}),

    dcc.Dropdown(id='slct_Target2', options=[
        {'label': i, 'value': i} for i in df['SK_ID_CURR']
    ],
                 multi=False,
                 value='CODE_GENDER',
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


#
# # ------------------------------------------------------------------------------
# # Connect the Plotly graphs with Dash Components
# @app.callback(
#     [Output(component_id='output_container', component_property='children'),
#      Output(component_id='graphique', component_property='figure')],
#     [Input(component_id='slct_Target', component_property='value')]
# )
# def update_graph(option_slctd):
#     print(option_slctd)
#     print(type(option_slctd))
#
#     container = "The year chosen by user was: {}".format(option_slctd)
#
#     dff = df.copy()
#     dff = dff[dff["TARGET"] == option_slctd]
#
#     # Plotly Express
#     fig = px.scatter(data_frame=dff,x='AMT_ANNUITY',y='AMT_GOODS_PRICE',template='plotly_dark')
#     return container, fig


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

    container = "{}".format(option_slctd)
    dff = df.copy()
    # Plotly Express
    # mydict={'Vous':(int(df[df['SK_ID_CURR']==option_slctd]['AMT_INCOME_TOTAL'])),'Moyenne des Demandeurs':(int(df[df['SK_ID_CURR']==option_slctd]['Mean_AMT_INCOME_TOTAL']))}
    langs = ['Vous', 'Moyenne Revenues Totales']
    students = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL']),
                int(df[df['SK_ID_CURR'] == option_slctd]['Mean_AMT_INCOME_TOTAL'])]
    fig1 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')
    fig2 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')
    fig3 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')
    fig4 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')
    fig = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')

    return container, fig,fig1,fig2,fig3,fig4


if __name__ == '__main__':
    app.run_server(debug=True)
