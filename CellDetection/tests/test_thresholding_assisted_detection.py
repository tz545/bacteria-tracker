import numpy as np
from collections import deque
from thresholding_assisted_detection import Shape, add_shapes_from_pixels, threshold_segmentation, segmentation_to_shapes, kmeans_segmentation
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def test_add_shapes_from_pixels():
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

	cells = add_shapes_from_pixels(points_set, cutoff=1)

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

	shape = Shape(set([tuple(x) for x in points]), mask)

	assert shape.points == set([tuple(x) for x in eroded_points1])
	assert shape.points == set([tuple(x) for x in eroded_points2])
	

def test_threshold_segmentation_whole_image():
	image = np.arange(9).reshape(3,3)
	mask1 = np.array([[1,2],[2,0],[2,1],[2,2]])

	mask2 = np.array([[1,1],[1,2],[2,0],[2,1],[2,2]])

	assert np.array_equal(threshold_segmentation(image, 1.1), mask1)
	assert np.array_equal(threshold_segmentation(image, 0.9), mask2)


def test_threshold_segmentation_set_of_points():
	image = np.arange(9).reshape(3,3)
	points = set([(1,0), (1,2), (2,0), (2,2)])

	mask1 = np.array([[2,0],[2,2]])
	mask2 = np.array([[2,2]])

	assert np.array_equal(threshold_segmentation(image, 1.0, points), mask1)
	assert np.array_equal(threshold_segmentation(image, 1.1, points), mask2)


def test_segmentation_to_shapes_adds_to_empty_cells():
	image = np.arange(9).reshape(3,3)
	mask1 = np.array([[1,2],[2,0],[2,1],[2,2]])

	cells = {}
	segmentation_to_shapes(cells, image, threshold_segmentation, 1.1, cutoff=1, smooth=False)

	assert len(cells) == 1
	assert cells[1].points == set([tuple(x) for x in mask1])


def test_segmentation_to_shapes_adds_to_existing_cells():
	image = np.arange(9).reshape(3,3)
	mask1 = np.array([[1,2],[2,0],[2,1],[2,2]])

	cells = {1:'a'}
	segmentation_to_shapes(cells, image, threshold_segmentation, 1.1, cutoff=1, smooth=False)

	assert len(cells) == 2
	assert cells[1] == 'a'
	assert cells[2].points == set([tuple(x) for x in mask1])


def test_segmentation_to_shapes_adds_several_shapes():
	image = np.arange(9).reshape(3,3)
	image[2,2] = 0
	image[0,2] = 8
	shape1 = set([(1,2),(0,2)])
	shape2 = set([(2,0), (2,1)])

	cells = {10:'a'}
	segmentation_to_shapes(cells, image, threshold_segmentation, 1.1, cutoff=1, smooth=False)

	assert len(cells) == 3
	assert cells[10] == 'a'
	assert cells[11].points == shape1
	assert cells[12].points == shape2


def test_segmentation_to_shapes_kmeans_incorporation():
	image = np.zeros([3,3], dtype=np.float32)
	image[2,1] = 1
	image[2,2] = 1

	cells = {10:'a'}
	segmentation_to_shapes(cells, image, kmeans_segmentation, cutoff=1, smooth=False)
	assert len(cells) > 1
	assert cells[10] == 'a'