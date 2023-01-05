import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, callback
import plotly.express as px
import plotly.graph_objects as go
import json
import base64
import os, pathlib

from app_functions import *

dash.register_page(__name__)

current_folder = pathlib.Path(__file__).parent.resolve()
dash_folder = current_folder.parent.absolute()
src_folder = dash_folder.parent.absolute()
project_folder = src_folder.parent.absolute()

layout = html.Div(children=[ 
    html.H3(children='Firing Detection', style={"margin-left":"30px", "margin-top":"20px"}),

    html.Div([
        dcc.Upload(id='upload-image', children=html.Div([
            html.A('Select Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    
    dcc.Upload(id='upload-boundaries', children=html.Div([
            html.A('Select Cell Tracking File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    )       
            ], style={"margin-left":"10px",'width': '48%', 'display': 'inline-block'}),

    html.Br(),
    html.Div(children=html.Button(id='button-confirm', n_clicks=0, children='Upload files'), style={"margin-left": "500px",'display': 'inline-block'}),

    html.Br(),

    html.Div(children=[
        html.H6(children="Threshold for firing (ratio of average intensity to maximum intensity)"),
        dcc.Slider(0,1,step=None,value=0.5,id='threshold-slider'),
        dcc.Graph(id="firing-visualization")],style={"margin-left":"30px","margin-top":"30px",'width': '40%','display': 'inline-block'}),

    html.Br(),
    html.Div(children=html.Div(id='image-stack-no-display'), style={"margin-left": "150px",'display': 'inline-block'}),

    html.Br(),
    html.Div(children=[
        html.Button(id='button-prev-f', n_clicks=0, children='Previous'),
        html.Button(id='button-next-f', n_clicks=0, children='Next', style={"margin-left":"10px"}),
        ], style={"margin-left": "150px",'display': 'inline-block'}),

    html.Br(),
    html.Details([
        html.Summary('Contents of output file:', style={"margin-left":"30px"}),
        dcc.Markdown(
            id='print-output'
        ),

    html.Br(),
    html.Div([html.Button("Download output", id="btn-download-output"), dcc.Download(id="download-output")], style={"margin-left": "150px",'display': 'inline-block'}),

    dcc.Store(id='raw-image-f'), 
    dcc.Store(id='num-frames-f'), 
    dcc.Store(id='boundary-info', storage_type='session'),
    dcc.Store(id='image-stack-no-f', data=0, storage_type='session'),
    dcc.Store(id='output-data', storage_type='session')
    ])

])

@callback(
    Output('image-stack-no-f', 'data'),
    Input('button-prev-f', 'n_clicks'),
    Input('button-next-f', 'n_clicks'),
    State('image-stack-no-f', 'data'),
    State('num-frames-f', 'data'), prevent_initial_call=True
    )
def update_frame_number(next, prev, stack_no, num_frames):
    ## check if at the end of the stack, if so, give save file option
    if ctx.triggered_id == 'button-next-f' and stack_no <= num_frames-3:
        return stack_no + 1

    elif ctx.triggered_id == 'button-prev-f' and stack_no >= 1:
        return stack_no - 1

    else:
        return stack_no

@callback(
    Output('image-stack-no-display', 'children'),
    Input('image-stack-no-f', 'data'), 
    Input('num-frames-f', 'data'), prevent_initial_call=True
    )
def update_frame_no_display_1(stack_no, num_frames):
    return 'Image Frame Number: {0}/{1}'.format(stack_no, num_frames-1)


@callback(
    Output('boundary-info', 'data'),
    Input('upload-boundaries', 'contents'), prevent_initial_call=True
)
def update_boundaries(boundaries_file):
    """Decodes binary string from json file and returns dictionary"""
    contents_list = boundaries_file.split(',')[1:]
    contents = ','.join(contents_list)
    decoded = base64.b64decode(contents)
    decoded = decoded.decode()
    cells_stack = json.loads(decoded)
    return cells_stack


@callback(
    Output('raw-image-f', 'data'),
    Output('num-frames-f', 'data'),
    Input('upload-image', 'filename'), prevent_initial_call=True
)
def update_image(image_file_name):
    image_file_path = os.path.join(project_folder, 'data', 'raw')
    image_file_name = os.path.join(image_file_path, image_file_name)
    image = process_image(image_file_name)
    image_list = image.tolist()
    return {'image': image, 'frames':len(image)}, len(image)


@callback(
    Output('firing-visualization', 'figure'),
    Input('button-confirm', 'n_clicks'),
    Input('image-stack-no-f', 'data'),
    Input('threshold-slider', 'value'),
    State('raw-image-f', 'data'),
    State('boundary-info', 'data'), prevent_initial_call=True
    )
def update_figure(confirm, stack_no, threshold, raw_image, cell_stack):

    raw_image = np.array(raw_image['image'], dtype=np.float32)
    raw_image = raw_image[stack_no]

    fig = px.imshow(raw_image,color_continuous_scale='gray', width=700, height=700) #color_continuous_scale='gray',
    fig.layout.coloraxis.showscale = False

    cells = cell_stack[str(stack_no)]

    for c in cells.keys():
        edges = np.array(cells[c]['boundary'])
        intensity_ratio = cells[c]['ave_intensity'] / cells[c]['max_intensity']
        if intensity_ratio <= threshold:
            fig.add_trace(go.Scatter(x=edges[::5,1], y=edges[::5,0], mode='lines', name='cell{0}'.format(c), line={'width':1, 'color':'rgb(255,0,0)'}, showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=edges[::5,1], y=edges[::5,0], mode='lines', name='cell{0}'.format(c), line={'width':1, 'color':'rgb(0,0,0)'}, showlegend=False))
    return fig
    

@callback(
    Output('output-data', 'data'),
    Input('threshold-slider', 'value'), 
    Input('button-confirm', 'n_clicks'),
    State('boundary-info', 'data'),
    State('num-frames-f', 'data'), prevent_initial_call=True
)
def calc_output(threshold, confirm, cells_stack, num_frames):
    firing_durations = []
    firing_cells = {}
    all_cells = set()
    for i in range(num_frames):
        cells = cells_stack[str(i)]
        new_cells = set(cells.keys())
        if len(all_cells) != len(new_cells):
            all_cells = all_cells.union(new_cells)

        for c in all_cells:
            if c in new_cells:
                if cells[c]['ave_intensity']/cells[c]['max_intensity'] <= threshold:
                    if c in firing_cells:
                        firing_cells[c] += 1
                    else:
                        firing_cells[c] = 1
                else:
                    if c in firing_cells:
                        firing_durations.append(firing_cells.pop(c))
            else:
                if c in firing_cells:
                    firing_durations.append(firing_cells.pop(c))

    for c in firing_cells:
        firing_durations.append(firing_cells.pop(c))

    return [{"no_firings": len(firing_durations), "durations": firing_durations, "no_cells": len(all_cells)}]


@callback(
    Output('print-output', 'children'),
    Input('output-data', 'data'), prevent_initial_call=True
)
def show_output(output):
    return '```\n'+json.dumps(output, indent=2)+'\n```'


@callback(
    Output("download-output", "data"),
    Input("btn-download-output", "n_clicks"),
    State('output-data', 'data'), prevent_initial_call=True,
)
def download(n_clicks, output):

    return dict(content=json.dumps(output), filename="firing_analysis.txt")
