import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from mypython import p7_1,p7_2

app = dash.Dash(__name__, suppress_callback_exceptions=True)
df = pd.read_csv("df_P7Clean.csv")

df['Ratio-GP-Annuity'] = ((df['AMT_INCOME_TOTAL']) / (df['AMT_CREDIT']))
df["Monthly"] = df['AMT_ANNUITY'] / 12
df["Mean_AMT_INCOME_TOTAL"] = np.mean(df['AMT_INCOME_TOTAL'])
df["Mean_Monthly"] = np.mean(df['Monthly'])
df["Mean_ANNUITY"] = np.mean(df['AMT_ANNUITY'])
df["Mean_AMT_CREDIT"] = np.mean(df['AMT_CREDIT'])
df["Mean_AMT_GOODS_PRICE"] = np.mean(df['AMT_GOODS_PRICE'])

app.layout = html.Div([

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
    print( ' coordeoner' + str(int(df[df['SK_ID_CURR'] == option_slctd]['AMT_INCOME_TOTAL'])))
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
    fig3 = px.histogram(data_frame=df, x=df['AMT_GOODS_PRICE'], template='plotly_dark',log_x=True)
    fig3.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['AMT_GOODS_PRICE']),line_dash="dash", line_color="red")
    fig3.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['AMT_GOODS_PRICE'],y=0,showarrow=False,text='<b>Vous</b>',
                            textangle=0,arrowcolor='red')
    fig4 = px.histogram(data_frame=df, x=df['AMT_ANNUITY'], template='plotly_dark',log_x=True)
    fig4.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['AMT_ANNUITY']),line_dash="dash", line_color="red")
    fig4.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['AMT_ANNUITY'],y=0,showarrow=False,text='<b>Vous</b>',
                            textangle=0,arrowcolor='red')
    fig = px.histogram(data_frame=df, x=df['Ratio-GP-Annuity'], template='plotly_dark',log_x=True)
    fig.add_vline(x=int(df[df['SK_ID_CURR'] == option_slctd]['Ratio-GP-Annuity']),line_dash="dash", line_color="red")
    fig.add_annotation(x=df[df['SK_ID_CURR'] == option_slctd]['Ratio-GP-Annuity'],y=0,showarrow=False,text='<b>Vous</b>',
                            textangle=0,arrowcolor='red')
    return container, fig, fig1, fig2, fig3, fig4


if __name__ == '__main__':
    app.run_server(debug=True)
