# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import pandas as pd
from scipy.spatial import Delaunay
from skimage import io

from unet import UNet
from cells_manipulation import Shape, mask_to_cells

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
            dcc.Dropdown(models_df['Description'].unique(), "None", id='model-choice-1'),
            dcc.Graph(id="cell-segmentation-1")],style={'width': '40%','display': 'inline-block'}),
        html.Div(children=[html.H4(children="Select model:"),
            dcc.Dropdown(models_df['Description'].unique(), "None", id='model-choice-2'),
            dcc.Graph(id="cell-segmentation-2")], style={"margin-left": "150px",'width': '40%','display': 'inline-block'})
    ]),
    html.Div([
        "Image Stack Number: ",
        dcc.Input(id='image-stack-no', value=0, type='number')
    ], style={"margin-left": "150px",'width': '40%','display': 'inline-block'}),
    dcc.Store(id='raw-image'), 
    dcc.Store(id='cells-1'),
    dcc.Store(id='cells-2')
])

def process_image(image_file):

    im = io.imread(image_file)
    im = im.astype(np.float32)

    ## take the middle quarter of images across all stacks
    ## (this cut is made because I have not figured out a good way of zooming in online)
    quadrant = im[:,im.shape[1]//4:im.shape[1]//2,im.shape[2]//4:im.shape[2]//2]
    quadrant = quadrant - np.min(quadrant)
    quadrant = quadrant/np.max(quadrant)

    return quadrant


def in_hull(p, hull):
    """
    https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl#:~:text=First%2C%20obtain%20the%20convex%20hull,clockwise%20around%20the%20convex%20hull.
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def add_cell(cells, fig_shape, lasso_select):

    lasso_dict = lasso_select['lassoPoints']
    lasso_points = np.column_stack([np.array(lasso_dict['y']), np.array(lasso_dict['x'])])
    
    ## check all pixels in rectangle bounding selection to see if they are within selection
    bottom_left = np.floor(np.min(lasso_points, axis=0)).astype(int)
    upper_right = np.ceil(np.max(lasso_points, axis=0)).astype(int)
    x_range = np.arange(max(0,bottom_left[0]), min(upper_right[0], fig_shape[0]))
    y_range = np.arange(max(0,bottom_left[1]), min(upper_right[1], fig_shape[1]))
    X, Y = np.meshgrid(x_range, y_range)
    pixel_grid_points = np.column_stack([X.ravel(), Y.ravel()])

    pixels_in_selection = in_hull(pixel_grid_points, lasso_points)
    new_shape = pixel_grid_points[pixels_in_selection==True]
    cells[max(cells.keys())+1] = Shape(set([tuple(x) for x in new_shape])).to_dict()

    return cells


def remove_cell(cells, mouse_click):
    point = mouse_click['points'][0]
    row = int(np.rint(point['y']))
    col = int(np.rint(point['x']))

    for c in cells.keys():
        if [row, col] in cells[c]['points']:
            cells.pop(c)
            break

    return cells


def general_update_cells(model_file_name, mouse_click, lasso_select, stack_no, raw_image, cells, model_choice_id):

    raw_image = np.array(raw_image['image'], dtype=np.float32)
    raw_image = raw_image[stack_no]

    if ctx.triggered_id is None or ctx.triggered_id == model_choice_id or ctx.triggered_id == "image-stack-no":
        if model_file_name is not None:
            model_file = models_df[models_df['Description']==model_file_name].iloc[0].File

            model = UNet(1, 4)
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
            model.eval()

        image = torch.from_numpy(raw_image)
        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)

        mask = torch.argmax(model(image), dim=1)
        mask = mask.squeeze().detach().numpy()

        cells = mask_to_cells(mask, return_dict=True)

    if mouse_click is not None:

        cells = remove_cell(cells, mouse_click)

    if lasso_select is not None:
        if 'lassoPoints' in lasso_select:
            cells = {int(k):v for k,v in cells.items()}
            cells = add_cell(cells, raw_image.shape, lasso_select)

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
    Input('upload-image', 'filename')
    )
def update_image(image_file_name):
    if image_file_name == None:
        image_file_name = "PA_vipA_mnG_15nN_20x20_16x16_1S_10ums_8_GFP-1-small.tif"

    image_file_name = 'cells_images/' + image_file_name
    image = process_image(image_file_name)
    image_list = image.tolist()
    return {
        'image': image
    }


@app.callback(
    Output('cells-1', 'data'),
    Input('model-choice-1', 'value'),
    Input('cell-segmentation-1', 'clickData'),
    Input('cell-segmentation-1', 'selectedData'),
    Input('image-stack-no', 'value'),
    State('raw-image', 'data'),
    State('cells-1', 'data'), prevent_initial_call=True
    )
def update_cells_1(model_file_name, mouse_click, lasso_select, stack_no, raw_image, cells):

    return general_update_cells(model_file_name, mouse_click, lasso_select, stack_no, raw_image, cells, "model-choice-1")


@app.callback(
    Output('cell-segmentation-1', 'figure'),
    Input('raw-image', 'data'),
    Input('image-stack-no', 'value'),
    Input('cells-1', 'data'),
    State('cell-segmentation-1', 'figure')
    )
def update_figure_1(raw_image, stack_no, cells, fig):

    return general_update_figure(raw_image, stack_no, cells, fig)
    

@app.callback(
    Output('cells-2', 'data'),
    Input('model-choice-2', 'value'),
    Input('cell-segmentation-2', 'clickData'),
    Input('cell-segmentation-2', 'selectedData'),
    Input('image-stack-no', 'value'),
    State('raw-image', 'data'),
    State('cells-2', 'data'), prevent_initial_call=True
    )
def update_cells_2(model_file_name, mouse_click, lasso_select, stack_no, raw_image, cells):

    return general_update_cells(model_file_name, mouse_click, lasso_select, stack_no+1, raw_image, cells, "model-choice-2")


@app.callback(
    Output('cell-segmentation-2', 'figure'),
    Input('raw-image', 'data'),
    Input('image-stack-no', 'value'),
    Input('cells-2', 'data'),
    State('cell-segmentation-2', 'figure')
    )
def update_figure_2(raw_image, stack_no, cells, fig):

    return general_update_figure(raw_image, stack_no+1, cells, fig)


if __name__ == '__main__':
    app.run_server(debug=True)
