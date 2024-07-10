from dash import Dash, dcc, html, Input, Output, callback
import os
import forecast as fc


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H1('Sentiment Analysis App'),
    html.Br(),
    html.I("Try typing a review about a movie ..."),
    html.Br(),
    dcc.Input(id="input1", type="text", placeholder="type here", debounce=True),
    html.Div(id='display-value')
])

@callback(Output('display-value', 'children'), Input('input1', 'value'))
def display_value(value):
    sentiment_val = fc.sentiment(value)
    return f'{sentiment_val}'

if __name__ == '__main__':
    app.run(debug=True)