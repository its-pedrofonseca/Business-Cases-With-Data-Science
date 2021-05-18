import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px

from app import app
from app import server
import home

# https://htmlcheatsheet.com/css/

######################################################Data##############################################################

df = pd.read_csv("C:/Users/migue/Desktop/data_preprocess.csv")
df["Year"] = df["Date"].str.split("-").str[0]

######################################################Interactive Components############################################

points_of_sale = ["1", "2", "3", "4"]

pos_options = [dict(label='' + pos, value=pos) for pos in points_of_sale]

pos_dropdown = dcc.Dropdown(
    id='pos_drop',
    options=pos_options,
    value='1',
    persistence=True,
    persistence_type='session'
)

quarters = ["1", "2", "3", "4"]

quarters_options = [dict(label='' + quarter, value=quarter) for quarter in quarters]

quarters_dropdown = dcc.Dropdown(
    id='quarters_drop',
    options=quarters_options,
    value='1',
    persistence=True,
    persistence_type='session'
)

years = ["2016", "2017", "2018", "2019"]

years_options = [dict(label='' + year, value=year) for year in years]

years_dropdown = dcc.Dropdown(
    id='years_drop',
    options=years_options,
    value='2016',
    persistence=True,
    persistence_type='session'
)

##################################################APP###################################################################

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4('Select the PoS:', className="text-center")
        ], width=4),
        dbc.Col([
            html.H4('Select the quarter:', className="text-center")
        ], width=4),
        dbc.Col([
            html.H4('Select the year:', className="text-center")
        ], width=4)
    ]),
    dbc.Row([
        dbc.Col([
            pos_dropdown,
        ], width=4),
        dbc.Col([
            quarters_dropdown,
        ], width=4),
        dbc.Col([
            years_dropdown,
        ], width=4)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='graph_value', style={'height': 570}),
                body=True, color="#000000"
            )
        ], width={'size': 12})
    ], className="my-2"),

], fluid=True)


@app.callback(
    Output('graph_value', 'figure'),
    [Input('pos_drop', 'value'),
     Input('quarters_drop', 'value'),
     Input('years_drop', 'value')]
)
def graph_1(pos, quarter, year):
    df_1 = df.loc[df['Point-of-Sale_ID'] == int(pos)]
    df_1 = df_1[(df_1['Quarter'] == int(quarter)) & (df_1["Year"] == year)]
    df_1['Date'] = pd.to_datetime(df_1['Date'])
    df_1 = df_1.groupby(df_1['Date']).sum()

    fig = px.bar(df_1, x=df_1.index, y=df_1["Units"], title='Value in Millions for thear')
    return fig
