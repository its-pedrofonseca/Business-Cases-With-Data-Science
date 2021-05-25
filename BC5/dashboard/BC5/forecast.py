import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX



from app import app
from app import server
import EDA

######################################################Data##############################################################

fdf = pd.read_csv('C:/Users/migue/Desktop/forecast_df.csv')
fdf['week'] = pd.to_datetime(fdf['week'], format='%Y-%m-%d')

weekly=fdf.groupby('week')['Units'].sum()[1:-1]
forecast = weekly.asfreq(pd.infer_freq(weekly.index))
start_date = datetime(2016,1,1)
end_date = datetime(2019,11,1)
lim_df = forecast[start_date:end_date]
dates = lim_df.index

######################################################Interactive Components############################################
pos_options = [dict(label=pos, value=pos) for pos in fdf['Point-of-Sale_ID'].dropna().unique()]
pid_options = [dict(label=pid, value=pid) for pid in fdf['ProductName_ID'].dropna().unique()]

######################################################Layout############################################
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Forecast by Point-of-Sale', className='text-center ')
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H6("Point-Of-Sale Choice", className='text-left text mb-4 ml-4'),
            dcc.Dropdown(
                id='pos_drop',
                options=pos_options,
                value=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                multi=True)
        ], width=12, className="mb-3"),
        dbc.Col([
            html.H6("Product Choice", className='text-left text mb-4 ml-4'),
            dcc.Dropdown(
                id='pid_drop',
                options=pos_options,
                value=[292,2609,2802,420],
                multi=True)
        ], width=12, className="mb-3")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='fig_forecast_all'), body=True, color="#31343b"
            )
        ], width={'size': 10}, className="mb-5 mt-3"),
        dbc.Col([
            dbc.Row([
                dbc.Col([dbc.Card([
                    dbc.CardBody([
                        html.H4('MAPE', className='text-white',style={'text-align': 'center'}),
                        dbc.ListGroup([
                            dbc.ListGroupItem(id='mape_all',style={'text-align': 'center'})
                        ])
                    ])
                ],color="#31343b"),],width=12)
            ],className="mb-2"),
            dbc.Row([
                dbc.Col([dbc.Card([
                    dbc.CardBody([
                        html.H4('RMSE', className='text-white',style={'text-align': 'center'}),
                        dbc.ListGroup([
                            dbc.ListGroupItem(id='rmse_all',style={'text-align': 'center'})
                        ])
                    ])
                ],color="#31343b"),],width=12)
            ],className="my-2"),
        ], width={'size': 2}, className="mb-5 mt-3"),
    ]),
], fluid=True)

@app.callback(
    Output('fig_forecast_all', 'figure'),
    [
        Input('pos_drop', 'value'),
        Input('pid_drop', 'value')
    ]
)
def forecast_by_POS_and_PID(pos, pid):
    # Setting up the group by on the ProductName_ID
    grouped = fdf[fdf['ProductName_ID'].isin(pid)][fdf['Point-of-Sale_ID'].isin(pos)].groupby('week')['Units'].sum()
    grouped = pd.Series(grouped, index=dates).fillna(0)
    # removing the first and last date for lack of values
    weekly = grouped[1:-1]

    # Infer the frequency of the data
    forecast = weekly.asfreq(pd.infer_freq(weekly.index))
    # Set DF
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2019, 11, 1)
    lim_df = forecast[start_date:end_date]

    # Set Train and Test Values
    train_end = datetime(2019, 8, 30)
    test_end = datetime(2019, 11, 1)
    train_data = lim_df[:train_end]
    test_data = lim_df[train_end + timedelta(days=1):test_end]

    zeros_prediction = test_data.copy().isna().replace(to_replace=False, value=0)
    # SETTING PREDICTIONS FOR ALMOST DISCONTINUED ITEMS TO 0
    if train_data[datetime(2019, 1, 1):].mean() < 1:
        predictions = zeros_prediction
    else:
        # SARIMA
        my_order = (1, 1, 0)  # (p,d,q) (AR,I,MA)
        my_seasonal_order = (0, 1, 1, 52)  #
        # Define model
        model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
        model_fit = model.fit()
        # Get the predictions and residuals
        predictions = model_fit.forecast(steps=len(test_data))
        predictions = pd.Series(predictions, index=test_data.index)
    residuals = test_data - predictions

    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=lim_df.index, y=lim_df,
                             mode='lines',
                             name='TimeSeries'))
    fig_all.add_trace(go.Scatter(x=predictions.index, y=predictions,
                             mode='lines',
                             name='Predictions'))

    return fig_all

@app.callback(
    [
        Output('mape_all', 'children'),
        Output('rmse_all', 'children'),
    ],
    [
        Input('pos_drop', 'value'),
        Input('pid_drop', 'value')
    ]
)
def kpi_by_POS_and_PID(pid, pos):
    # Setting up the group by on the ProductName_ID and POS_ID
    grouped = fdf[fdf['ProductName_ID'].isin(pid)][fdf['Point-of-Sale_ID'].isin(pos)].groupby('week')['Units'].sum()
    grouped = pd.Series(grouped, index=dates).fillna(0)
    # removing the first and last date for lack of values
    weekly = grouped[1:-1]

    # -----------// FORECASTING //-----------#
    # Infer the frequency of the data
    forecast = weekly.asfreq(pd.infer_freq(weekly.index))
    # Set DF
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2019, 11, 1)
    lim_df = forecast[start_date:end_date]
    # Get First Diferences to eliminate Trend
    first_diff = lim_df.diff()[1:]

    # Set Train and Test Values
    train_end = datetime(2019, 8, 30)
    test_end = datetime(2019, 11, 1)
    train_data = lim_df[:train_end]
    test_data = lim_df[train_end + timedelta(days=1):test_end]

    # SARIMA
    my_order = (0, 1, 0)  # (p,d,q) (AR,I,MA)
    my_seasonal_order = (0, 1, 1, 52)  #
    # Define model
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()
    # Get the predictions and residuals
    predictions = model_fit.forecast(steps=len(test_data))
    predictions = pd.Series(predictions, index=test_data.index)
    residuals = test_data - predictions

    MAPE = round(np.mean(abs(residuals / test_data)), 4)
    RMSE = round(np.sqrt(np.mean(residuals ** 2)), 4)

    return str(MAPE),\
           str(RMSE)