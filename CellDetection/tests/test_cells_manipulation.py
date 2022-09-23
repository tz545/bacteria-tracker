import numpy as np 
from cells_manipulation import Shape, pixels_to_shapes, mask_to_cells
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def test_pixels_to_shapes():
	mask = np.zeros([11, 11], dtype=int)
	
	## not big enough to be a shape
	mask[0,0] = 1

	shape1 = (0,0)

	## big enough to be a shape
	mask[1,1:5] = 1
	mask[2,1:5] = 1
	shape2 = {(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4)}

	## also big enough to be a shape
	mask[6,9:] = 1
	mask[7,:10] = 1
	shape3 = {(6,9),(6,10),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9)}

	points = np.argwhere(mask==1)
	points_set = set([tuple(x) for x in points])

	cells = pixels_to_shapes(points_set, cutoff=1)

	assert len(cells) == 2
	assert shape1 not in cells
	assert shape2 in cells
	assert shape3 in cells
	

def test_shape_smooth_shape():
	mask = np.zeros([11, 11], dtype=int)
	points = np.array([[3,4],[3,5],[4,4],[5,3],[5,4],[5,5],[6,4],[6,5]])
	mask[points[:,0], points[:,1]] = 1
	kernel = np.ones([3,3], dtype=int)
	dilation = binary_dilation(mask, kernel).astype(int)
	erosion = binary_erosion(dilation, kernel).astype(int)
	eroded_points1 = np.argwhere(erosion==1)
	eroded_points2 = np.array([[3,4],[3,5],[4,4],[4,5],[5,3],[5,4],[5,5],[6,4],[6,5]])

	shape = Shape(set([tuple(x) for x in points]), mask, smooth=True)

	assert shape.points == set([tuple(x) for x in eroded_points1])
	assert shape.points == set([tuple(x) for x in eroded_points2])
	

def test_mask_to_shapes_handles_overlaps():

	a = np.zeros([4,4])
	a[0,1] = 1
	a[1,1] = 2
	a[1,2] = 2
	a[2,2] = 1
	a[3,2] = 2
	a[3,3] = 1

	cells_list = mask_to_cells(a, a, 0, 10)

	shape1 = Shape(set([(0,1), (1,1), (1,2)]), a)
	shape2 = Shape(set([(1,2), (3,2), (1,1), (2,2)]), a)
	shape3 = Shape(set([(3,2), (3,3)]), a)

	assert len(cells_list) == 3

	cells_list_points = [x.points for x in cells_list.values()]

	assert shape1.points in cells_list_points
	assert shape2.points in cells_list_points
	assert shape3.points in cells_list_points


