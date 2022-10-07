import numpy as np
from thresholding_assisted_detection import threshold_segmentation, segmentation_to_shapes, kmeans_segmentation

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
	segmentation_to_shapes(cells, threshold_segmentation, 1.1, cutoff=1)

	assert len(cells) == 1
	assert cells[1].points == set([tuple(x) for x in mask1])


def test_segmentation_to_shapes_adds_to_existing_cells():
	image = np.arange(9).reshape(3,3)
	mask1 = np.array([[1,2],[2,0],[2,1],[2,2]])

	cells = {1:'a'}
	segmentation_to_shapes(cells, threshold_segmentation, 1.1, cutoff=1)

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
	segmentation_to_shapes(cells, threshold_segmentation, 1.1, cutoff=1)

	assert len(cells) == 3
	assert cells[10] == 'a'
	assert cells[11].points == shape1
	assert cells[12].points == shape2


def test_segmentation_to_shapes_kmeans_incorporation():
	image = np.zeros([3,3], dtype=np.float32)
	image[2,1] = 1
	image[2,2] = 1

	cells = {10:'a'}
	segmentation_to_shapes(cells, kmeans_segmentation, cutoff=1)
	assert len(cells) > 1
	assert cells[10] == 'a'