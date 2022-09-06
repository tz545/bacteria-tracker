import numpy as np
from collections import deque
from CellDetection.still_image_object_detection import Shape, binary_mask_to_shapes

def test_binary_mask_to_shapes():
	mask = np.zeros([11, 11], dtype=int)
	
	## not big enough to be a shape
	mask[0,0] = 1
	shape1 = Shape({(0,0)})

	## big enough to be a shape
	mask[1,1:5] = 1
	mask[2,1:5] = 1
	shape2 = Shape({(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4)})

	## also big enough to be a shape
	mask[6,9:] = 1
	mask[7,:10] = 1
	shape3 = Shape({(6,9),(6,10),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9)})

	cells = binary_mask_to_shapes(mask, cutoff=1)

	cells_list = list(cells.values())
	cells_points_list = [x.points for x in cells_list]

	assert len(cells_list) == 2
	assert shape1.points not in cells_points_list
	assert shape2.points in cells_points_list
	assert shape3.points in cells_points_list
	

