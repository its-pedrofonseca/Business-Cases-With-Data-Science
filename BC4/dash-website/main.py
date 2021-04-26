import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Connect to main index.py file
from app import app
from app import server

logo = dbc.Navbar(
    dbc.Container(
        [

            html.A(
                dbc.Row(
                    [

                        html.Img(src=app.get_asset_url("MGUKbar.png"), height="70px",
                                         className="mr-auto"),
                    ],
                    align="center",
                    className='align-self-center',
                    no_gutters=True,
                ),
            ),
        ], fluid=True,
    ),

    color="#1450a0",
    dark=True,
    className="mb-2 mr-0",
)

content = html.Div(id="page-content")

app.layout = html.Div(
    [dcc.Location(id="url"), logo, content],
)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
