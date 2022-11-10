# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

from processing_functions import *

app = Dash(__name__)

models_df = pd.DataFrame({
    "Description": ["High Sensitivity", "Good Generality", "None"],
    "File": ["models/first_image/15_epochs.pt", "models/low_contrast_expanded_dataset_sweep/50_epochs_lr_0.0005_m_0.1_best.pt", None]
    })

app.layout = html.Div(children=[
    html.H3(children='Cell Detection'),

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
    )
            ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div(children=[
        html.Div(children=[html.H4(children="Select model:"),
            dcc.Dropdown(models_df['Description'].unique(), "High Sensitivity", id='model-choice-1'),
            dcc.Graph(id="cell-segmentation-1")],style={'width': '40%','display': 'inline-block'}),
        html.Div(children=[html.H4(children="Select model:"),
            dcc.Dropdown(models_df['Description'].unique(), "High Sensitivity", id='model-choice-2'),
            dcc.Graph(id="cell-segmentation-2")], style={"margin-left": "150px",'width': '40%','display': 'inline-block'})
    ]),
    html.Div(children=[html.Div(id='image-stack-no-display-1', style={"margin-left": "150px",'display': 'inline-block'}),
                    html.Div(id='image-stack-no-display-2', style={"margin-left": "600px",'display': 'inline-block'})]),

    html.Br(),
    html.Div(children=[
        html.Button(id='button-prev', n_clicks=0, children='Previous'),
        html.Button(id='button-save', n_clicks=0, children='Track cells in next frame'),
        html.Button(id='button-next', n_clicks=0, children='Next'),
        html.Div([html.Button("Save cells", id="btn-download-cells"), dcc.Download(id="download-cells")])
        ], style={"margin-left": "150px",'display': 'inline-block'}),

    dcc.Store(id='raw-image'), 
    dcc.Store(id='num-frames'), 
    dcc.Store(id='cells', storage_type='session'),
    dcc.Store(id='image-stack-no', data=0, storage_type='memory'),
    dcc.Store(id='temp-cells-1', storage_type='local'),
    dcc.Store(id='temp-cells-2', storage_type='session')
])


def general_update_cells(trig, model_file_name, mouse_click, lasso_select, stack_no, raw_image, cells, model_choice_id):

    if trig == model_choice_id or trig == 'button-next':
        if model_file_name is not None:
            model_file = models_df[models_df['Description']==model_file_name].iloc[0].File

            model = UNet(1, 4)
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
            model.eval()

        raw_image = np.array(raw_image['image'], dtype=np.float32)
        raw_image = raw_image[stack_no]

        image = torch.from_numpy(raw_image)
        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)

        mask = torch.argmax(model(image), dim=1)
        mask = mask.squeeze().detach().numpy()

        cells = mask_to_cells(mask, return_dict=True)

    if mouse_click is not None:

        result = remove_cell(cells, mouse_click)
        if result is not None:
            print("cell deleted")
            cells = result

    if lasso_select is not None:

        raw_image = np.array(raw_image['image'], dtype=np.float32)
        raw_image = raw_image[stack_no]

        if 'lassoPoints' in lasso_select:
            cells = {int(k):v for k,v in cells.items()}
            cells = add_cell(cells, raw_image.shape, lasso_select)

            print("cell added")

    return cells


def general_update_figure(raw_image, stack_no, cells, fig):

    raw_image = np.array(raw_image['image'], dtype=np.float32)
    raw_image = raw_image[stack_no]

    fig = px.imshow(raw_image,color_continuous_scale='gray', width=700, height=700) #color_continuous_scale='gray',
    fig.layout.coloraxis.showscale = False

    if cells is not None:

        for c in cells.keys():
            edges = np.array(cells[c]['boundary'])
            ## only draw every 5 points on each boundary to optimize speed
            fig.add_trace(go.Scatter(x=edges[::5,1], y=edges[::5,0], mode='lines', name='cell{0}'.format(c), line={'width':1}, showlegend=False))
            fig.add_trace(go.Scatter(x=[cells[c]['center'][1]], y=[cells[c]['center'][0]], mode='markers', name='cell{0}'.format(c), marker={'color':'rgb(255,255,255)', 'size':4}, showlegend=False))

    return fig


@app.callback(
    Output('raw-image', 'data'),
    Output('num-frames', 'data'),
    Input('upload-image', 'filename')
    )
def update_image(image_file_name):
    if image_file_name == None:
        image_file_name = "PA_vipA_mnG_15nN_20x20_16x16_1S_10ums_8_GFP-1-small.tif"

    image_file_name = 'cells_images/' + image_file_name
    image = process_image(image_file_name)
    image_list = image.tolist()
    return {'image': image, 'frames':len(image)}, len(image)


@app.callback(
    Output('image-stack-no-display-1', 'children'),
    Input('image-stack-no', 'data'), 
    Input('num-frames', 'data')
    )
def update_frame_no_display_1(stack_no, num_frames):
    return 'Image Frame Number: {0}/{1}'.format(stack_no, num_frames-1)


@app.callback(
    Output('image-stack-no-display-2', 'children'),
    Input('image-stack-no', 'data'), 
    Input('num-frames', 'data')
    )
def update_frame_no_display_2(stack_no, num_frames):
    return 'Image Frame Number: {0}/{1}'.format(stack_no+1, num_frames-1)


@app.callback(
    Output('image-stack-no', 'data'),
    Input('button-prev', 'n_clicks'),
    Input('button-next', 'n_clicks'),
    State('image-stack-no', 'data'),
    State('num-frames', 'data'), prevent_initial_call=True
    )
def update_frame_number(next, prev, stack_no, num_frames):
    ## check if at the end of the stack, if so, give save file option
    if ctx.triggered_id == 'button-next' and stack_no <= num_frames-3:
        return stack_no + 1

    elif ctx.triggered_id == 'button-prev' and stack_no >= 1:
        return stack_no - 1

    else:
        return stack_no


@app.callback(
    Output("download-cells", "data"),
    Input("btn-download-cells", "n_clicks"),
    State('cells', 'data'), prevent_initial_call=True,
)
def download(n_clicks, cells):
    return dict(content=json.dumps(cells), filename="cells.txt")


@app.callback(
    Output('temp-cells-1', 'data'),
    Input('model-choice-1', 'value'),
    Input('cell-segmentation-1', 'clickData'),
    Input('cell-segmentation-1', 'selectedData'),
    Input('button-prev', 'n_clicks'),
    Input('button-next', 'n_clicks'),
    State('image-stack-no', 'data'),
    State('raw-image', 'data'),
    State('temp-cells-1', 'data'),
    State('temp-cells-2', 'data'),
    State('cells', 'data'), prevent_initial_call=True
    )
def update_cells_1(model_file_name, mouse_click, lasso_select, n_prev, n_next, stack_no, raw_image, cells1, cells2, saved_cells):

    ## check here as well if end of stack reached
    if ctx.triggered_id == 'button-next':
        if stack_no >= raw_image['frames'] - 2:
            return cells1
        else:
            return cells2

    elif ctx.triggered_id == 'button-prev':
        if stack_no >= 1:
            return saved_cells[str(stack_no-1)]
        else: 
            return cells1

    else:
        return general_update_cells(ctx.triggered_id, model_file_name, mouse_click, lasso_select, stack_no, raw_image, cells1, "model-choice-1")


@app.callback(
    Output('cell-segmentation-1', 'figure'),
    Input('raw-image', 'data'),
    Input('image-stack-no', 'data'),
    Input('temp-cells-1', 'data'),
    State('cell-segmentation-1', 'figure')
    )
def update_figure_1(raw_image, stack_no, cells, fig):

    return general_update_figure(raw_image, stack_no, cells, fig)
    

@app.callback(
    Output('temp-cells-2', 'data'),
    Input('model-choice-2', 'value'),
    Input('cell-segmentation-2', 'clickData'),
    Input('cell-segmentation-2', 'selectedData'),
    Input('button-prev', 'n_clicks'),
    Input('button-save', 'n_clicks'),
    Input('button-next', 'n_clicks'),
    State('image-stack-no', 'data'),
    State('raw-image', 'data'),
    State('temp-cells-1', 'data'),
    State('temp-cells-2', 'data'),
    State('cells', 'data'), prevent_initial_call=True
    )
def update_cells_2(model_file_name, mouse_click, lasso_select, n_prev, n_save, n_next, stack_no, raw_image, cells1, cells2, saved_cells):

    if ctx.triggered_id == 'button-save':
        ## propagate cells from first frame onto second frame
        return forward_prop_cells(cells1, cells2)

    elif ctx.triggered_id == 'button-prev':
        if stack_no >= 1:
            return cells1
        else:
            return cells2

    elif ctx.triggered_id == 'button-next' and stack_no+1 in saved_cells.keys():
        return saved_cells[str(stack_no+1)]

    elif ctx.triggered_id == 'button-next' and stack_no >= raw_image['frames'] - 2:
        return cells2

    else:
        return general_update_cells(ctx.triggered_id, model_file_name, mouse_click, lasso_select, stack_no+1, raw_image, cells2, "model-choice-2")


@app.callback(
    Output('cell-segmentation-2', 'figure'),
    Input('raw-image', 'data'),
    Input('image-stack-no', 'data'),
    Input('temp-cells-2', 'data'),
    State('cell-segmentation-2', 'figure')
    )
def update_figure_2(raw_image, stack_no, cells, fig):

    return general_update_figure(raw_image, stack_no+1, cells, fig)


@app.callback(
    Output('cells', 'data'),
    Input('button-save', 'n_clicks'),
    Input('button-next', 'n_clicks'),
    State('temp-cells-1', 'data'),
    State('temp-cells-2', 'data'),
    State('image-stack-no', 'data'),
    State('cells', 'data'), prevent_initial_call=True
    )
def save_cells(n_save, n_next, temp_cells1, temp_cells2, stack_no, cells):
    
    if ctx.triggered_id == 'button-save':
        if stack_no == 0:
            cells = {}
        cells[stack_no] = temp_cells1

    elif ctx.triggered_id == 'button-next':
        cells[stack_no+1] = temp_cells2

    return cells


if __name__ == '__main__':
    app.run_server(debug=True)
