import numpy as np 
import matplotlib.pyplot as plt
import cv2

from cells_manipulation import Shape, CellSplitter, pixels_to_shapes, manual_correction


def kmeans_segmentation(image, points=None):
	"""Computes segmentation using k=2 (cells/background) on either entire image
	or subset of image pixels (as given by set of points)
	https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/?ref=rp"""

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
	k = 2

	if points == None:
		retval, labels, _ = cv2.kmeans(image.flatten(), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	else:
		points_array = np.array(list(points))
		im_vals = image[points_array[:,0], points_array[:,1]]
		retval, labels, _ = cv2.kmeans(im_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	centers = np.uint8([[0],[1]])
	segmented_data = centers[labels.flatten()]

	if points == None:
		mask = segmented_data.reshape((image.shape))
		return np.argwhere(mask==1)

	else:
		return points_array[segmented_data.flatten()==1]


def threshold_segmentation(image, threshold, points=None):
	"""Computes binary threshold segmentation on either entire image
	or subset of image pixels (as given by set of points)
	threshold: multiplier of median value used for segmentation"""

	if points == None:
		binary_mask = np.sign(image - threshold*np.median(image)).astype(int)
		return np.argwhere(binary_mask==1)
		
	else:
		points_array = np.array(list(points))
		im_vals = image[points_array[:,0], points_array[:,1]]
		binary_mask = np.sign(im_vals - threshold*np.median(im_vals)).astype(int)	
		return points_array[binary_mask==1]


def segmentation_to_shapes(image, cells, segmentation_function, *threshold, points=None, cutoff=4):
	"""Either an identified region needs to be split into multiple cells
	or an unselected region contains cells that need to be segmented
	image: 2D numpy array of pixel values
	points: set of (tuple) pixel coordinates that will undergo another round of segmentation"""

	segmented_points = segmentation_function(image, *threshold, points)

	shape_pixels_set = set([tuple(x) for x in segmented_points])

	new_shapes_list = pixels_to_shapes(shape_pixels_set, cutoff)

	for s in new_shapes_list:
		if len(cells) == 0:
			cells[len(cells)+1] = Shape(s)
		else:
			cells[max(cells.keys())+1] = Shape(s)



def threshold_segment_images(image_file, image_number):
	"""Full threshold segmentation pipeline.
	Loads an image file, splits into quadrants, applies adaptive thresholding,
	allows user correction and saves image and mask

	image_number: number that first quadrant in image is saved under"""

	im = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
	imarray = np.array(im.reshape(im.shape), dtype=np.float32)

	assert imarray.shape[0] % 2 == 0 and imarray.shape[1] % 2 == 0, "Input image dimensions should be even."
	quadrants = [imarray[:imarray.shape[0]//2, :imarray.shape[0]//2],\
				imarray[imarray.shape[0]//2:, :imarray.shape[0]//2],\
				imarray[:imarray.shape[0]//2, imarray.shape[0]//2:],\
				imarray[imarray.shape[0]//2:, imarray.shape[0]//2:]]

	for q in range(len(quadrants)):

		## Blur the image to get rid of patchy artifacts
		img_blur = cv2.GaussianBlur(quadrants[q],(3,3), sigmaX=10, sigmaY=10)

		## rescale so the values fit into uint8 (required for adaptiveThreshold function)
		img_blur = (img_blur - min(img_blur.flatten()))/max(img_blur.flatten()) * 255
		img_blur = img_blur.astype('uint8')

		grey2 = cv2.adaptiveThreshold(src=img_blur, dst=img_blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=7, C=min(img_blur.flatten()))
		img_blur = cv2.GaussianBlur(img_blur.astype(np.float32),(11,11), sigmaX=10, sigmaY=10)

		cells = {}

		segmentation_to_shapes(img_blur, cells, threshold_segmentation, 1.5, cutoff=100)

		image_with_mask = manual_correction(cells, quadrants[q])

		np.save('data/cell_{0}.npy'.format(image_number+q), image_with_mask)



if __name__ == "__main__":

	threshold_segment_images('cells_images/PA_vipA_mnG_30x30_32x32_35nN_75uNs_2s_1_GFP-1.tif', 8)
	

	
	

	

	

	







