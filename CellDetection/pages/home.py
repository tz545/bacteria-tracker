import dash
from dash import html, dcc

dash.register_page(__name__, path='/')

layout = html.Div(children=[
    html.H3(children='Usage basics:'),

    html.Div(children=['''
        Boundary detection: performs the semi-automated detection of cells and tracks them across frames. 
         
    ''', 
    html.Br(),
    '''Firing detection: detects firing cells using an adjustable thresholding of the ratio between the average and maximum intensities within a cell.
    ''']),

])