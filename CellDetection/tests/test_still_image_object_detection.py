import numpy as np
from collections import deque
from still_image_object_detection import Shape, binary_mask_to_shapes
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def test_binary_mask_to_shapes():
	mask = np.zeros([11, 11], dtype=int)
	
	## not big enough to be a shape
	mask[0,0] = 1
	shape1 = Shape({(0,0)}, (11,11), smooth=False)

	## big enough to be a shape
	mask[1,1:5] = 1
	mask[2,1:5] = 1
	shape2 = Shape({(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4)}, (11,11), smooth=False)

	## also big enough to be a shape
	mask[6,9:] = 1
	mask[7,:10] = 1
	shape3 = Shape({(6,9),(6,10),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9)}, (11,11), smooth=False)

	cells = binary_mask_to_shapes(mask, (11,11), cutoff=1, smooth=False)

	cells_list = list(cells.values())
	cells_points_list = [x.points for x in cells_list]

	assert len(cells_list) == 2
	assert shape1.points not in cells_points_list
	assert shape2.points in cells_points_list
	assert shape3.points in cells_points_list
	

def test_shape_smooth_shape():
	mask = np.zeros([11, 11], dtype=int)
	points = np.array([[3,4],[3,5],[4,4],[5,3],[5,4],[5,5],[6,4],[6,5]])
	mask[points[:,0], points[:,1]] = 1
	kernel = np.ones([3,3], dtype=int)
	dilation = binary_dilation(mask, kernel).astype(int)
	erosion = binary_erosion(dilation, kernel).astype(int)
	eroded_points1 = np.argwhere(erosion==1)
	eroded_points2 = np.array([[3,4],[3,5],[4,4],[4,5],[5,3],[5,4],[5,5],[6,4],[6,5]])

	shape = Shape(set([tuple(x) for x in points]), (11,11))

	assert shape.points == set([tuple(x) for x in eroded_points1])
	assert shape.points == set([tuple(x) for x in eroded_points2])
	

