from dash import Dash, html, dcc 
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

load_figure_template('SIMPLEX')

app = Dash(__name__, use_pages=True,external_stylesheets=[dbc.themes.SIMPLEX])

app.layout = html.Div([
	html.H1('Detecting T6SS harpoon firings in P. aeruginosa',style={"margin-left": "10px","margin-top":"20px", "margin-bottom":"20px"}),

    html.Div(
        [
            html.Div(
                dcc.Link(
                    f"{page['name']} - {page['path']}", href=page["relative_path"],style={"margin-left": "10px", "margin-right":"30px"}
                )
            )
            for page in dash.page_registry.values()
        ]
    ),

	dash.page_container
])

if __name__ == '__main__':
	app.run_server(debug=True)