import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.morphology import binary_dilation
#from boundary_edges import alpha_shape
from collections import deque


class Shape():


	def __init__(self, points):
		self.points = self.smooth_shape(points) # set of indices (corresponding to original image) within Shape
		self.boundary = self.calc_boundary()
		self.size = self.calc_size()

	def calc_boundary(self):
		## convex hull
		pass

	def calc_size(self):
		pass

	def smooth_shape(self, points):
		## we want to fill in small holes and get rid of rough edges
		## convert indices list to matrix
		## do dilation followed by erosion
		return points

	def cursor_in_shape(self, cursor_click):
		## check if clicked pixel is within dictionary of shape points
		pass

	def delete_shape(self):
		pass


	def segment_shape(self):
		## apply watershed image segregation to split shape further 
		pass



def raw_image_to_bindary_mask(image, segmentation, show_mask=False):
	"""Initial segmentation of raw image. Segmentation options are:
	'threshold', using 1.5x the median brightness value, or
	'kmeans', using k-means clustering with k=2"""

	im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
	imarray = np.array(im.reshape([2048,2048]), dtype=np.float32)

	## Blur the image to get rid of patchy artifacts
	img_blur = cv2.GaussianBlur(imarray,(3,3), sigmaX=0, sigmaY=0)

	if segmentation == 'threshold':
		segmented_image = np.sign(img_blur-1.5*np.median(img_blur.flatten()))
		segmented_image = segmented_image.astype(int)
		segmented_image[segmented_image==-1] = 0

	elif segmentation == 'kmeans':
		## https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/?ref=rp
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
		k = 2
		retval, labels, centers = cv2.kmeans(img_blur.flatten(), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
		 
		## construct binary mask using labels
		centers = np.uint8([[0],[1]])
		segmented_data = centers[labels.flatten()]
		segmented_image = segmented_data.reshape((img_blur.shape))

	if show_mask:

		## Use binary dilation to get rough boundaries of segmentation for visualization only
		## https://stackoverflow.com/questions/51696326/extracting-boundary-of-a-numpy-array
		kernel = np.ones((3,3),dtype=int) # for 4-connected
		segmented_boundaries = binary_dilation(segmented_image==0, kernel) & segmented_image

		boundary_mask = np.zeros([2048,2048,4])
		boundary_mask[:,:,-1] = segmented_boundaries

		plt.imshow(imarray)
		plt.imshow(boundary_mask)
		plt.show()



def binary_mask_to_shapes(mask, cutoff=4):
	"""Initial segmentation step returns mask of 0s (background) and 1s (shapes).
	This function converts all groups of connected pixels (bigger than the cutoff size) 
	into Shape objects and stores them within a dictionary."""

	## np.argwhere(x==1) returns array of [row,column] 
	shape_pixels = np.argwhere(mask==1)
	shape_pixels_set = set([tuple(x) for x in shape_pixels])

	cells = dict()
	cell_no = 0

	## BFS to identify neighbours  
	while len(shape_pixels_set) > 0:

		new_shape = set()
		root_pixel = shape_pixels_set.pop()

		BFS_queue = deque()
		BFS_queue.append(root_pixel)

		while len(BFS_queue) > 0:

			pixel = BFS_queue.popleft()
			new_shape.add(pixel)

			## search for 4-neighbours of the pixel
			row = pixel[0]
			col = pixel[1]

			neighboring_pixels = [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]
			for p in neighboring_pixels:
				if p in shape_pixels_set:
					shape_pixels_set.remove(p)
					BFS_queue.append(p)

		## we want to apply a cutoff to remove unconnected dots
		if len(new_shape) > cutoff:
			cell_no += 1
			cells[cell_no] = Shape(new_shape)

	return cells



if __name__ == "__main__":

	raw_image_to_bindary_mask('cells.tif', segmentation='kmeans', show_mask=True)






