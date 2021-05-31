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
import datetime as dt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import networkx as nx

from app import app
from app import server
import EDA

######################################################Data##############################################################

df = pd.read_csv("C:/Users/Pedro/Desktop/Business Cases/BC5/Datasets/mba_dash.csv")

######################################################Interactive Components############################################

points_of_sale = ["1", "2", "3", "4", '5',   '6',   '7',   '8',   '9',  '10',  '11',  '12',  '13',
        '14',  '15',  '16',  '17',  '18',  '19',  '20']

pos_options = [dict(label='' + pos, value=pos) for pos in points_of_sale]

pos_dropdown = dcc.Dropdown(
    id='pos_drop',
    options=pos_options,
    value='1',
    persistence=True,
    persistence_type='session'
)

quarters = ["1", "2", "3"]

quarters_options = [dict(label='' + quarter, value=quarter) for quarter in quarters]

quarters_dropdown = dcc.Dropdown(
    id='quarters_drop',
    options=quarters_options,
    value='1',
    persistence=True,
    persistence_type='session'
)

years = ["2016"]

years_options = [dict(label='' + year, value=year) for year in years]

years_dropdown = dcc.Dropdown(
    id='years_drop',
    options=years_options,
    value='2016',
    persistence=True,
    persistence_type='session'
)

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
        ], width=4),
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
        ], width=4),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='graph_value_mba', style={'height': 580}),
                body=True, color="#000000"
            )
        ], width={'size': 12}),
    ], className="my-2"),


], fluid=True)

@app.callback(
    Output('graph_value_mba', 'figure'),
    [Input('pos_drop', 'value'),
     Input('quarters_drop', 'value'),
     Input('years_drop', 'value'),]
)
def graph_1(pos, quarter, year):

    #filtering and operations
    orders = df[(df['Year'] == year) & (df['Quarter'] == quarter) & (df['Point-of-Sale_ID'] == pos)]

    pt = pd.pivot_table(orders[['TID', 'ProductName_ID']], index='TID', columns='ProductName_ID',
                        aggfunc=lambda x: 1 if len(x) > 0 else 0).fillna(0)

    frequent_itemsets = apriori(pt, min_support=0.05, use_colnames=True)

    rulesLift = association_rules(frequent_itemsets, metric="lift", min_threshold=0)

    rulesLift.sort_values(by='confidence', ascending=False, inplace=True)

    #drawing the plot
    data_scatter = dict(type='scatter',
                        y=rulesLift['confidence'],
                        x=rulesLift['lift'],
                        # text=rulesLift.index,
                        # mode='markers',
                        # marker=dict(
                        # size=rulesLift['support'],
                        hovertemplate=  # 'Grand Prix: ' + df_racetracks["name_x"] + '<br>'
                        # 'RuleID: ' + rulesLift.index + '<br>'+
                        'Lift: ' + rulesLift["lift"].astype(str) + '<br>' +
                        'Confidence: ' + rulesLift['confidence'].astype(str) + '<br>' +
                        'Support: ' + rulesLift['support'].astype(str) + '<br>' +
                        'Antecedents: ' + rulesLift['antecedents'].astype(str) + '<br>' +
                        'Consequents: ' + rulesLift['consequents'].astype(str) + '<br>'
                                                                                 '<extra></extra>',
                        # color=scatterdf['Avg Salary'],  # set color equal to a variable
                        # color_continuous_scale='mint',  # one of plotly colorscales
                        # showscale=False,
                        mode='markers',
                        marker=dict(size=8,
                                    # size=rulesLift['support'],
                                    color=rulesLift['support'],
                                    colorscale='oranges',
                                    showscale=True,
                                    line_width=2),
                        )

    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Lift",
        yaxis_title="Confidence"
    )

    fig = go.Figure(data=data_scatter, layout=layout)

    return fig