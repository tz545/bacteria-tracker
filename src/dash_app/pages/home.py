import dash
from dash import html, dcc

dash.register_page(__name__, path='/')

layout = html.Div(children=[
    html.H3(children='Usage instructions:',style={"margin-left": "30px", "margin-right":"30px", "margin-top":"20px"}),

    html.Div(children=[dcc.Markdown('''
    #### Boundary Detection
    *This page is for manually correcting the cell detection algorithm and tracking cells between consecutive frames.*  
      
    1. Select image from file (currently images must be saved in bacteria-tracker/data/raw/).  
    2. Select one model option for each image. "High sensitivity" and "Good generality" are two pre-trained ML models. "None" implments simple threshold-based segmentation.  
    3. Make edits to the left image:   
        a. To remove a cell, single click on the outline or white dot.  
        b. To add a cell, drag and click to zoom into a region, then use the lasso select tool (from the toolbox on the upper right corner of the image) to mark out the new boundary. The app will automatically zoom out after each lasso selection.  
    ''',style={"margin-left": "30px", "margin-right":"30px", "margin-top":"20px"}),
    html.Img(src='assets/lasso_select.png', height='250', style={"margin-left": "80px"}),
    dcc.Markdown('''
    4. When the left image is perfect, remove all the incorrectly identified cells on the right image (single-click on their outlines). Clicking the "Track cells in next frame" button below the left image should identify the remainder correctly if they haven't moved much. If this fails for individual cells, remove the incorrect cells from the right image, zoom in and label the cell correctly, and try again with the "Track cells in next frame". If this fails, remove the cell and redraw directly. A newly-drawn cell will take on the number of the cell that was deleted immediately prior. At this point, the colours and numbers of the cells should be consistent between the left and right panels, as shown below. 
    ''',style={"margin-left": "30px", "margin-right":"30px", "margin-top":"10px"}),
    html.Img(src='assets/cell_tracking.png', height='280', style={"margin-left": "100px"}),
    dcc.Markdown('''
    5. Press "Next". The right image now becomes the left image, and repeat Step 4. At each step, ensure that cell colours and numbers align after pressing "Track cells in next frame".   
    6. Repeat Step 5 until the end of the image stack is reached.    
    7. Download the cell tracking file by clicking the "Save cells" button.  
    ''',style={"margin-left": "30px", "margin-right":"30px"}),
    html.Br(),
    dcc.Markdown('''
    #### Firing Detection
    *This page detects firing events using an adjustable thresholding of the ratio of the average intensity to the maximum intensity within a cell.*  
      
    1. Select image and cell tracking file. Image must still be under bacteria-tracker/data/raw/. The cell tracking file can be saved anywhere.   
    2. Click "Upload files".  
    3. Use the "Previous" and "Next" buttons under "Image Frame Number" to flip through the frames of the image. Cells that determined to be not firing (as per the threshold) are indicated with a black outline. Firing cells are indicated with a red outline. Adjust the slider to change the threshold for firing as needed. 
    ''',style={"margin-left": "30px", "margin-right":"30px"}),
    html.Img(src='assets/threshold_adjust.png', height='400', style={"margin-left": "100px"}),
    dcc.Markdown('''
    4. "Contents of output file" expands to show the number of firing events (across all cells and all frames), the durations of each of the firing events (in numbers of frames), and the total number of cells within the image. This can be downloaded as a text file using "Download output".
    ''',style={"margin-left": "30px", "margin-right":"30px", "margin-top":"10px"})
    ]),
 
])