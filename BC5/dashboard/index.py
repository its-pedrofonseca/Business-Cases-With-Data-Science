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

import EDA
import MBA
import forecast
from app import app

nav_item_EDA = dbc.NavItem(dbc.NavLink("EDA", href="/EDA", active="exact"))
nav_item_MBA = dbc.NavItem(dbc.NavLink("MBA", href="/MBA", active="exact"))
nav_item_forecast = dbc.NavItem(dbc.NavLink("Forecast", href="/forecast", active="exact"))


CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

logo = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        html.Img(src=app.get_asset_url("img.png"), height="70px",
                                 className="mr-auto"),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://www.formula1.com/",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    [nav_item_EDA, nav_item_MBA, nav_item_forecast], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ],fluid=True
    ),
    color="#31343b",
    dark=True,
    className="mb-3",
)

content = html.Div(id="page-content")

app.layout = html.Div(
    [dcc.Location(id="url"), logo, content],
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return EDA.layout
    if pathname == "/EDA":
        return EDA.layout
    elif pathname == "/MBA":
        return MBA.layout
    elif pathname == "/forecast":
        return forecast.layout
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == '__main__':
    app.run_server(debug=True)