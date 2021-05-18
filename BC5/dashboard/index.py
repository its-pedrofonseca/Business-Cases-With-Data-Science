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

import home
from app import app

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
                    className='align-self-center',
                    no_gutters=True,
                ),
            ),
        ], fluid=True,
    ),

    color="#000000",
    dark=True,
    className="mb-2 mr-0",
)

content = html.Div(id="page-content")

app.layout = html.Div(
    [dcc.Location(id="url"), logo, content],
)

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return home.layout
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == '__main__':
    app.run_server(debug=True)