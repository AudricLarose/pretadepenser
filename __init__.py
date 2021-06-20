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
df['Ratio-GP-Annuity'] = ((df['AMT_INCOME_TOTAL']) / (df['AMT_CREDIT']))
df["Monthly"] = df['AMT_ANNUITY'] / 12
df["Mean_AMT_INCOME_TOTAL"] = np.mean(df['AMT_INCOME_TOTAL'])
df["Mean_Monthly"] = np.mean(df['Monthly'])
df["Mean_ANNUITY"] = np.mean(df['AMT_ANNUITY'])
df["Mean_AMT_CREDIT"] = np.mean(df['AMT_CREDIT'])
df["Mean_AMT_GOODS_PRICE"] = np.mean(df['AMT_GOODS_PRICE'])



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
    langs = ['Vous', 'Moyenne Revenues Totales']
    students = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL']),
                int(df[df['SK_ID_CURR'] == option_slctd]['Mean_AMT_INCOME_TOTAL'])]
    fig1 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')
    fig2 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')
    fig3 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')
    fig4 = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')
    fig = px.bar(data_frame=df, x=langs, y=students, template='plotly_dark')

    return container, fig,fig1,fig2,fig3,fig4

def create (flask_app) :
    dash_app = dash.Dash(
        server=flask_app,name="dashboard",url_base_pathname="/dash/"
    )
    dash_app.layout = html.Div([

        html.H1("Comparaison client vis a vis des autres demandeurs de crédit", style={'text-align': 'center'}),

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

    @dash_app.callback(
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
        you = ['Vous', 'Moyenne Revenues Totales']
        income = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL']),
                    int(df[df['SK_ID_CURR'] == option_slctd]['Mean_AMT_INCOME_TOTAL'])]
        monthly = [int(df[df['SK_ID_CURR'] == option_slctd]['Monthly']),
                  int(df[df['SK_ID_CURR'] == option_slctd]['Mean_Monthly'])]
        annuity = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_ANNUITY']),
                  int(df[df['SK_ID_CURR'] == option_slctd]['Mean_ANNUITY'])]
        credit = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_CREDIT']),
                  int(df[df['SK_ID_CURR'] == option_slctd]['Mean_AMT_CREDIT'])]
        good = [int(df[df['SK_ID_CURR'] == option_slctd]['AMT_GOODS_PRICE']),
                  int(df[df['SK_ID_CURR'] == option_slctd]['Mean_AMT_GOODS_PRICE'])]
        fig1 = px.bar(data_frame=df, x=you, y=income, color_discrete_sequence=['red'], template='plotly_dark',title='Comparaison vos Revenues vs la moyenne')
        fig2 = px.bar(data_frame=df, x=you, y=monthly, color_discrete_sequence=['blue'],template='plotly_dark',title='Comparaison mensualité vs la moyenne')
        fig3 = px.bar(data_frame=df, x=you, y=annuity,color_discrete_sequence=['green'], template='plotly_dark',title='Comparaison revenues mensuelles vs la moyenne')
        fig4 = px.bar(data_frame=df, x=you, y=credit,color_discrete_sequence=['orange'], template='plotly_dark',title='Comparaison credit vs la moyenne')
        fig = px.bar(data_frame=df, x=you, y=good, color_discrete_sequence=['yellow'],template='plotly_dark',title='Comparaison montant du bien vs la moyenne')
        return container, fig, fig1, fig2, fig3, fig4
    return dash_app



if __name__ == '__main__':
    app.run_server(debug=True)
