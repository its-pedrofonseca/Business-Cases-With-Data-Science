import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pathlib
from dash.dependencies import Input, Output
#from implicit.als import AlternatingLeastSquares


# Connect to main index.py file
from app import app
from app import server

PATH = pathlib.Path(__file__)
DATA_PATH = PATH.joinpath("../data/").resolve()
#-------------  Importing  ----------------
df = pd.read_csv(DATA_PATH.joinpath("dfdash.csv"))

#-------------  Interactive components ----------------
login_options = [dict(label=user, value=user) for user in df['CustomerID'].dropna().unique()]

#models
#alpha = 1
#als_model = AlternatingLeastSquares(factors=200, regularization=1e1, iterations=20, random_state=70)
#als_model.fit((df * alpha).astype('double'))


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
    [
        dcc.Location(id="url"), logo, content,
        dbc.Row(
            [
                dbc.Col([
                    html.H3("Login:", className='text-left text mb-4 ml-4'),
                ], width=2, className="mb-3"
                ),
                dbc.Col([
                    dcc.Dropdown(
                        id='login_drop',
                        options=login_options,
                        value=[17850],
                        # multi=True
                    )
                ], width=10, className="mb-3"
                ),
            ],
            align="center",
            className='align-self-center',
            no_gutters=True,
        ),
        dbc.Row(
            [
                dbc.Card([
                    dbc.CardBody([
                        html.H4('Some Clients also liked:', className='text-white'),
                        dbc.ListGroup([
                            dbc.ListGroupItem(id='lista')
                        ])
                    ])
                ], color="#1450a0", className='mb-5'),
            ],
            align="center",
            className='align-self-center',
            no_gutters=True,
        ),
        dbc.Row(
            [
                html.Img(src=app.get_asset_url("MGUKprod.png"),
                                         className="mr-auto"),
            ],
            align="center",
            className='align-self-center',
            no_gutters=True,
        ),
        ]
)

@app.callback(
    [
    Output('lista', 'children'),
    ],
    [
    Input('login_drop', 'value')
    ]
)

def recommend_items(user):
    description_list=[]
    numberTrans=list(df[df['CustomerID']==user].groupby('CustomerID')['TransactionId'].count())[0]
    if numberTrans < 11:
        print('Less than 10 Transactions, Popular Algorithm used')
        recommendations = pop_model.recommend(user_map[user_id],user_item_train)
        recommendations_list = list(map(lambda x: (get_keys(x[0], item_map), x[1]), recommendations))
        for i in range(len(recommendations_list)):
            description_list.append(df[df['ItemId']==recommendations_list[i][0]]['Description'].unique()[0])
        return description_list
    else:
        recommendations=als_model.recommend(user_map[user_id],user_item_train)
        recommendations_list =  list(map(lambda x: (get_keys(x[0], item_map), x[1]), recommendations))
        for i in range(len(recommendations_list)):
            description_list.append(df[df['ItemId']==recommendations_list[i][0]]['Description'].unique()[0])
        return description_list
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
