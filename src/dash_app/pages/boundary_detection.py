import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os, pathlib
import cv2

from app_functions import *

dash.register_page(__name__)

current_folder = pathlib.Path(__file__).parent.resolve()
dash_folder = current_folder.parent.absolute()
src_folder = dash_folder.parent.absolute()
project_folder = src_folder.parent.absolute()
model_folder = os.path.join(project_folder, "models")

models_df = pd.DataFrame({
    "Description": ["High Sensitivity", "Good Generality", "None"],
    "File": [os.path.join(model_folder, "unet_01.pt"), os.path.join(model_folder, "unet_02.pt"), None]
    })

layout = html.Div(children=[
    html.H3(children='Cell Detection', style={"margin-left": "30px", "margin-right":"30px", "margin-top":"20px"}),

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
            'margin': '30px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    )
            ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div(children=[
        html.Div(children=[html.H4(children="Select model:",style={"margin-left":'50px'}),
            dcc.Dropdown(models_df['Description'].unique(), "High Sensitivity", id='model-choice-1', style={"margin-left":'30px'}),
            dcc.Graph(id="cell-segmentation-1")],style={'width': '40%','display': 'inline-block'}),
        html.Div(children=[html.H4(children="Select model:",style={"margin-left":'110px'}),
            dcc.Dropdown(models_df['Description'].unique(), "High Sensitivity", id='model-choice-2',style={"margin-left":'60px'}),
            dcc.Graph(id="cell-segmentation-2")], style={"margin-left":'100px','width': '40%','display': 'inline-block'})
    ]),
    html.Div(children=[html.Div(id='image-stack-no-display-1', style={"margin-left": "50px",'display': 'inline-block'}),
                    html.Div(id='image-stack-no-display-2', style={"margin-left": "500px",'display': 'inline-block'})]),

    html.Br(),
    html.Div(children=[
        html.Button(id='button-prev', n_clicks=0, children='Previous'),
        html.Button(id='button-save', n_clicks=0, children='Track cells in next frame', style={"margin-left":"10px"}),
        html.Button(id='button-next', n_clicks=0, children='Next', style={"margin-left":"10px"}),
        html.Br(),
        html.Div([html.Button("Save cells", id="btn-download-cells"), dcc.Download(id="download-cells")])
        ], style={"margin-left": "150px", "margin-top":"50px", 'display': 'inline-block'}),

    dcc.Store(id='raw-image'), 
    dcc.Store(id='temp-image-left'),
    dcc.Store(id='temp-image-right'),
    dcc.Store(id='num-frames'), 
    dcc.Store(id='cells', storage_type='session'),
    dcc.Store(id='image-stack-no', data=0, storage_type='session'),
    dcc.Store(id='temp-cells-1', storage_type='local'),
    dcc.Store(id='temp-cells-2', storage_type='session'), 
    dcc.Store(id='last-mouse-click-1', storage_type='memory'),
    dcc.Store(id='last-mouse-click-2', storage_type='memory'),
    dcc.Store(id='last-cell-no-1', data=-1),
    dcc.Store(id='last-cell-no-2', data=-1)
])


def general_update_cells(trig, model_file_name, mouse_click, lasso_select, raw_image, cells, model_choice_id, last_click, last_no):

    if trig == model_choice_id or trig == 'button-next':

        raw_image = np.array(raw_image, dtype=np.float32)

        if model_file_name != 'None':
            ## use UNet to segment
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

        else:
            ## use threshold segmentation
            img_blur = cv2.GaussianBlur(raw_image,(3,3), sigmaX=10, sigmaY=10)
            img_blur = (img_blur - min(img_blur.flatten()))/max(img_blur.flatten()) * 255
            img_blur = img_blur.astype('uint8')
            grey2 = cv2.adaptiveThreshold(src=img_blur, dst=img_blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=7, C=min(img_blur.flatten()))
            img_blur = cv2.GaussianBlur(img_blur.astype(np.float32),(11,11), sigmaX=10, sigmaY=10)
            
            cells = segmentation_to_cells(img_blur, {}, threshold_segmentation, 1.5, cutoff=100, return_dict=True)

        return cells, last_click, last_no

    if lasso_select is not None:

        if 'lassoPoints' in lasso_select:

            raw_image = np.array(raw_image, dtype=np.float32)

            cells = {int(k):v for k,v in cells.items()}
            if last_no == -1:
                cells, last_no = add_cell(cells, raw_image.shape, lasso_select)
            else:
                cells, last_no = add_cell(cells, raw_image.shape, lasso_select, last_no)
            return cells, last_click, last_no

    if mouse_click is not None:

        if mouse_click != last_click:

            result = remove_cell(cells, mouse_click)
            if result is not None:
                cells, last_no = result

            last_click = mouse_click

    return cells, last_click, last_no


def general_update_figure(raw_image, cells, fig):

    raw_image = np.array(raw_image, dtype=np.float32)

    fig = px.imshow(raw_image,color_continuous_scale='gray', width=700, height=700) #color_continuous_scale='gray',
    fig.layout.coloraxis.showscale = False

    if cells is not None:

        for c in cells.keys():
            edges = np.array(cells[c]['boundary'])
            ## only draw every 5 points on each boundary to optimize speed
            fig.add_trace(go.Scatter(x=edges[::5,1], y=edges[::5,0], mode='lines', name='cell{0}'.format(c), line={'width':1}, showlegend=False))
            fig.add_trace(go.Scatter(x=[cells[c]['center'][1]], y=[cells[c]['center'][0]], mode='markers', name='cell{0}'.format(c), marker={'color':'rgb(255,255,255)', 'size':4}, showlegend=False))

    return fig


@callback(
    Output('raw-image', 'data'),
    Output('num-frames', 'data'),
    Input('upload-image', 'filename')
    )
def update_image(image_file_name):
    if image_file_name == None:
        image_file_name = "Test_file_1.tif"

    image_file_path = os.path.join(project_folder, 'data', 'raw')
    image_file_name = os.path.join(image_file_path, image_file_name)
    image = process_image(image_file_name)

    return {'image': image, 'frames':len(image)}, len(image)


@callback(
    Output('image-stack-no-display-1', 'children'),
    Output('image-stack-no-display-2', 'children'),
    Output('temp-image-left', 'data'),
    Output('temp-image-right', 'data'),
    Input('image-stack-no', 'data'), 
    Input('num-frames', 'data'), 
    State('raw-image', 'data')
    )
def update_frame_no_display(stack_no, num_frames, raw_image):
    raw_image = np.array(raw_image['image'], dtype=np.float32)
    return 'Image Frame Number: {0}/{1}'.format(stack_no, num_frames-1), 'Image Frame Number: {0}/{1}'.format(stack_no+1, num_frames-1), raw_image[stack_no], raw_image[stack_no + 1]

@callback(
    Output('image-stack-no', 'data'),
    Input('button-prev', 'n_clicks'),
    Input('button-next', 'n_clicks'),
    State('image-stack-no', 'data'),
    State('num-frames', 'data')
    )
def update_frame_number(next, prev, stack_no, num_frames):
    ## check if at the end of the stack, if so, give save file option
    if ctx.triggered_id == 'button-next' and stack_no <= num_frames-3:
        return stack_no + 1

    elif ctx.triggered_id == 'button-prev' and stack_no >= 1:
        return stack_no - 1

    else:
        return stack_no


@callback(
    Output("download-cells", "data"),
    Input("btn-download-cells", "n_clicks"),
    State('cells', 'data'),
    State('raw-image', 'data'),  prevent_initial_call=True,
)
def download(n_clicks, cells, raw_image):
    """saves cell boundaries, and compute and save average and max intensity"""

    raw_image = np.array(raw_image['image'], dtype=np.float32)

    file_info = {}

    ## need to iterate through each stack of images and cells
    for stack in range(len(cells)):
        raw_image_slice = raw_image[stack]
        cell_slice = cells[str(stack)]

        ## re-number cells so cell numbers are consecutive
        cell_indices = [int(k) for k in cell_slice.keys()]
        sorted_indices = np.sort(cell_indices)
        cell_map = dict(zip(sorted_indices, np.arange(len(sorted_indices))))

        new_cells = {}
        for k,v in cell_slice.items():
            points_array = np.array(cell_slice[k]['points'])
            im_vals = raw_image_slice[points_array[:,0], points_array[:,1]]
            new_cells[int(cell_map[int(k)])] = {'boundary':v['boundary'], 'max_intensity':float(np.max(im_vals)), 'ave_intensity':float(np.mean(im_vals))}

        file_info[stack] = new_cells

    return dict(content=json.dumps(file_info), filename="cells.json")


@callback(
    Output('temp-cells-1', 'data'),
    Output('last-mouse-click-1', 'data'),
    Output('last-cell-no-1', 'data'),
    Input('model-choice-1', 'value'),
    Input('cell-segmentation-1', 'clickData'),
    Input('cell-segmentation-1', 'selectedData'),
    State('image-stack-no', 'data'),
    State('num-frames', 'data'),
    State('temp-image-left', 'data'),
    State('temp-cells-1', 'data'),
    State('temp-cells-2', 'data'),
    State('cells', 'data'),
    State('last-mouse-click-1', 'data'),
    State('last-cell-no-1', 'data'), prevent_initial_call=True
    )
def update_cells_1(model_file_name, mouse_click, lasso_select, stack_no, num_frames, raw_image, cells1, cells2, saved_cells, last_click, last_cell_no):

    # print(ctx.triggered_id, mouse_click, lasso_select, flush=True)
    if ctx.triggered_id == 'button-next':
        if stack_no >= num_frames - 2:
            return cells1, last_click, last_cell_no
        else:
            return cells2, last_click, -1

    elif ctx.triggered_id == 'button-prev':
        if stack_no >= 1:
            return saved_cells[str(stack_no-1)], last_click, -1
        else: 
            return cells1, last_click, last_cell_no

    else:
        return general_update_cells(ctx.triggered_id, model_file_name, mouse_click, lasso_select, raw_image, cells1, "model-choice-1", last_click, last_cell_no)


@callback(
    Output('cell-segmentation-1', 'figure'),
    Input('temp-image-left', 'data'),
    Input('temp-cells-1', 'data'),
    State('cell-segmentation-1', 'figure')
    )
def update_figure_1(raw_image, cells, fig):

    return general_update_figure(raw_image, cells, fig)
    

@callback(
    Output('temp-cells-2', 'data'),
    Output('last-mouse-click-2', 'data'),
    Output('last-cell-no-2', 'data'),
    Input('model-choice-2', 'value'),
    Input('cell-segmentation-2', 'clickData'),
    Input('cell-segmentation-2', 'selectedData'),
    Input('button-prev', 'n_clicks'),
    Input('button-save', 'n_clicks'),
    Input('button-next', 'n_clicks'),
    State('image-stack-no', 'data'),
    State('num-frames', 'data'),
    State('temp-image-right', 'data'),
    State('temp-cells-1', 'data'),
    State('temp-cells-2', 'data'),
    State('cells', 'data'), 
    State('last-mouse-click-2', 'data'),
    State('last-cell-no-2', 'data'), prevent_initial_call=True
    )
def update_cells_2(model_file_name, mouse_click, lasso_select, n_prev, n_save, n_next, stack_no, num_frames, raw_image, cells1, cells2, saved_cells, last_click, last_cell_no):

    if ctx.triggered_id == 'button-save':
        ## propagate cells from first frame onto second frame
        return forward_prop_cells(cells1, cells2), last_click, -1

    elif ctx.triggered_id == 'button-prev':
        if stack_no >= 1:
            return cells1, last_click, -1
        else:
            return cells2, last_click, last_cell_no

    elif ctx.triggered_id == 'button-next' and stack_no+1 in saved_cells.keys():
        return saved_cells[str(stack_no+1)], last_click, -1

    elif ctx.triggered_id == 'button-next' and stack_no >= raw_image['frames'] - 2:
        return cells2, last_click, last_cell_no

    else:
        return general_update_cells(ctx.triggered_id, model_file_name, mouse_click, lasso_select, raw_image, cells2, "model-choice-2", last_click, last_cell_no)

@callback(
    Output('cell-segmentation-2', 'figure'),
    Input('temp-image-right', 'data'),
    Input('temp-cells-2', 'data'),
    State('cell-segmentation-2', 'figure')
    )
def update_figure_2(raw_image, cells, fig):

    return general_update_figure(raw_image, cells, fig)


@callback(
    Output('cells', 'data'),
    Input('button-save', 'n_clicks'),
    State('temp-cells-1', 'data'),
    State('temp-cells-2', 'data'),
    State('image-stack-no', 'data'),
    State('cells', 'data'), prevent_initial_call=True
    )
def save_cells(n_save, temp_cells1, temp_cells2, stack_no, cells):
    
    if stack_no == 0:
        cells = {}
    cells[str(stack_no)] = temp_cells1
    cells[str(stack_no+1)] = temp_cells2

    return cells
