import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import numpy as np
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pickle
import json
import pathlib
from app import server

server.layout = html.Div([
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

if __name__ == '__main__':
    server.run()
