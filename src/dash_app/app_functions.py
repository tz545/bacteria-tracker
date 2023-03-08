from scipy.spatial import Delaunay
from skimage import io
import numpy as np
import torch
from collections import deque
import sys, pathlib

current_folder = pathlib.Path(__file__).parent.resolve()
src_folder = str(current_folder.parent.absolute())
if src_folder not in sys.path:
    sys.path.insert(0, src_folder)

from utils import Shape, segmentation_to_cells, threshold_segmentation, mask_to_cells
from models.unet import UNet


def access_image(image_file, page=0):
    """Grab two consecutive frames from image file"""

    im = io.imread(image_file)
    im = im.astype(np.float32)

    im = im - np.min(im)
    im = im/np.max(im)

    return im[page:page+2], len(im)


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


def add_cell(cells, fig_shape, lasso_select, cell_no=None):

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
    new_no = max(cells.keys())+1

    ## allows new cell to be replaced by cell that was previously deleted
    if cell_no is None:
        cells[new_no] = Shape(set([tuple(x) for x in new_shape])).to_dict()
        new_no +=1
    else:
        cells[cell_no] = Shape(set([tuple(x) for x in new_shape])).to_dict()
    return cells, new_no


def remove_cell(cells, mouse_click):
    point = mouse_click['points'][0]
    row = point['y']
    col = point['x']

    for c in cells.keys():
        [x, y] = cells[c]['center']
        if ((row-x)**2 + (col-y)**2) <= 4:
            cells.pop(c)
            return cells, c

    return None


def forward_prop_cells(cell1, cell2):

	## convert cell2 dict to a mask of the image space, which we will match to cells1

	points_set = set()
	for k in cell2.keys():
		points_set.update([tuple(x) for x in cell2[k]['points']])

	new_cell2 = {}

	cell1 = {int(k):v for k,v in cell1.items()}
	for k in cell1.keys():
		cell_k = cell1[k]
		points_set_k = set([tuple(x) for x in cell_k['points']])
		common_points = points_set.intersection(points_set_k)

		## if cell practically disappears, keep the old cell
		if len(common_points)/cell_k['size'] < 0.2:
			new_cell2[k] = cell_k

		## if the cell is practically unchanged, keep the old cell
		elif len(common_points)/cell_k['size'] >= 0.8:
			new_cell2[k] = cell_k

		else:
			new_shape = set()
			BFS_queue = deque(common_points)
			while len(BFS_queue) > 0:
				pixel = BFS_queue.popleft()
				new_shape.add(pixel)

				## search for 4-neighbours of the pixel
				row = pixel[0]
				col = pixel[1]

				neighboring_pixels = [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]
				for p in neighboring_pixels:
					if p in points_set:
						points_set.remove(p)
						BFS_queue.append(p)

			new_cell2[k] = Shape(new_shape).to_dict()

	return new_cell2