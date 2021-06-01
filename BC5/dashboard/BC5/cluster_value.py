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
import EDA

df = pd.read_csv("C:/Users/bruno/OneDrive/Ambiente de Trabalho/Datasets/pca_value.csv")
value = pd.read_csv("C:/Users/bruno/OneDrive/Ambiente de Trabalho/Datasets/final_value.csv")
value.set_index('Point-of-Sale_ID',inplace=True)
df.set_index('Point-of-Sale_ID',inplace=True)

def scatter_plot():
    fig = px.scatter(df, x="PC0", y="PC1", color="cluster_value", hover_data=[df.index])
    return fig

def count_labels(df,label,columnToCount):
    d ={'Cluster':df.groupby(label)[columnToCount].count().index,'Count':df.groupby(label)[columnToCount].count().values}
    df =pd.DataFrame(data=d )
    fig = px.bar(df, x='Cluster', y='Count')
    return fig

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H4('Cluster 0', className='text-white',style={'text-align': 'center'}),
                    dbc.ListGroup([
                        dbc.ListGroupItem(round(value[value['cluster_value']==0].mean().mean()))
                    ],style={'text-align': 'center'})
                ])
            ,color="#31343b")
        ],width={'size':4}),

        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H4('Cluster 1', className='text-white', style={'text-align': 'center'}),
                    dbc.ListGroup([
                        dbc.ListGroupItem(round(value[value['cluster_value'] == 1].mean().mean()))
                    ],style={'text-align': 'center'})
                ])
                , color="#31343b")
        ], width={'size': 4}),

        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H4('Cluster 2', className='text-white', style={'text-align': 'center'}),
                    dbc.ListGroup([
                        dbc.ListGroupItem(round(value[value['cluster_value'] == 2].mean().mean()))
                    ],style={'text-align': 'center'})
                ])
                , color="#31343b")
        ], width={'size': 4}),

    ],className="mb-2"),

    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(figure =scatter_plot(),style={'height': 500}),
                body=True, color="#31343b"
            )
        ],width={'size': 6}),

        dbc.Col([
            dbc.Card(
                dcc.Graph(figure=count_labels(df,'cluster_value','PC0'), style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 6}),

    ])
], fluid=True)