import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from boundary_edges import alpha_shape, stitch_boundaries
from collections import deque


class Shape():

	def __init__(self, points, im, smooth=True):
		self.im = im # original grayscale values for further segmentation if needed
		self.imshape = im.shape # size of original image
		if smooth:
			self.points = self.smooth_shape(points) # set of pixel indices (corresponding to original image) within Shape
		else:
			self.points = points
		self._boundary = None
		self.size = len(self.points) # can consider automatic thresholding of sizes
		
	@property
	def boundary(self):
		if self._boundary is None:
			self.calc_boundary()
		return self._boundary
	
	def calc_boundary(self):
		"""Calculates the edges corresponding to the convex hull. 
			Does not give uninterrupted line of pixels which forms the boundary.
			Used only for drawing and identifying outlines of shapes."""

		points_array = np.array(list(self.points))
		edges = alpha_shape(points_array, alpha=1.0, only_outer=True)
		boundary_list = stitch_boundaries(edges)
		boundary_polygon = []
		for p in boundary_list[0]:
			boundary_polygon.append(points_array[p[0]])
		boundary_polygon.append(points_array[boundary_list[0][-1][1]])
			
		self._boundary = np.array(boundary_polygon)

	def smooth_shape(self, points):
		## we want to fill in small holes and get rid of rough edges
		
		## convert indices list to matrix
		points_array = np.array(list(points))
		row = points_array[:,0]
		col = points_array[:,1]

		mask = np.zeros(self.imshape, dtype=int)
		mask[row,col] = 1

		## do dilation followed by erosion
		kernel = np.ones([3,3], dtype=int)
		dilation = binary_dilation(mask, kernel).astype(int)
		erosion = binary_erosion(dilation, kernel).astype(int)

		pixels = np.argwhere(erosion==1)
		pixels_set = set([tuple(x) for x in pixels])

		return pixels_set

		

class CellSplitter:
	def __init__(self, fig, axes, cells):
		self.fig = fig
		self.axes = axes
		self.cells = cells
		self.cid = fig.canvas.mpl_connect('button_press_event', self)

	def __call__(self, event):
		print('click', event)
		if event.inaxes!=self.axes: return

		to_delete = []
		## double click on mouse to remove falsely identified cells
		if event.dblclick and event.button == 1:
			row = int(np.rint(event.ydata))
			col = int(np.rint(event.xdata))

			for c in cells.keys():
				if (row, col) in cells[c].points:
					cell_id_to_lines[c].remove()
					fig.canvas.draw()
					to_delete.append(c)
		
		for c in to_delete:
			self.cells.pop(c)
			cell_id_to_lines.pop(c)


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


def add_shapes_from_pixels(pixels, cutoff):
	"""pixels: set of (tuple) pixel coordinates"""

	new_shapes_list = []

	## BFS to identify neighbours  
	while len(pixels) > 0:

		new_shape = set()
		root_pixel = pixels.pop()

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
				if p in pixels:
					pixels.remove(p)
					BFS_queue.append(p)

		## we want to apply a cutoff to remove unconnected dots
		if len(new_shape) > cutoff:
			new_shapes_list.append(new_shape)

	return new_shapes_list


def segmentation(cells, image, points=None, cutoff=4, smooth=True):
	"""Either an identified region needs to be split into multiple cells
	or an unselected region contains cells that need to be segmented
	image: 2D numpy array of pixel values
	points: set of (tuple) pixel coordinates that will undergo another round of kmeans segmentation"""

	segmented_points = kmeans_segmentation(image, points)

	shape_pixels_set = set([tuple(x) for x in segmented_points])

	new_shapes_list = add_shapes_from_pixels(shape_pixels_set, cutoff)

	for s in new_shapes_list:
		cells[len(cells)+1] = Shape(s, im, smooth)

	return cells



# def raw_image_to_binary_mask(image, segmentation, show_mask=False):
# 	"""Initial segmentation of raw image. Segmentation options are:
# 	'threshold', using 1.5x the median brightness value, or
# 	'kmeans', using k-means clustering with k=2"""

# 	im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
# 	imarray = np.array(im.reshape(im.shape), dtype=np.float32)

# 	## Blur the image to get rid of patchy artifacts
# 	img_blur = cv2.GaussianBlur(imarray,(3,3), sigmaX=0, sigmaY=0)

# 	if segmentation == 'threshold':
# 		segmented_image = np.sign(img_blur-1.5*np.median(img_blur.flatten()))
# 		segmented_image = segmented_image.astype(int)
# 		segmented_image[segmented_image==-1] = 0

# 	elif segmentation == 'kmeans':	
# 		segmented_image = kmeans_segmentation(img_blur)

# 	if show_mask:

# 		## Use binary dilation to get rough boundaries of segmentation for visualization only
# 		## https://stackoverflow.com/questions/51696326/extracting-boundary-of-a-numpy-array
# 		kernel = np.ones((3,3),dtype=int) # for 4-connected
# 		segmented_boundaries = binary_dilation(segmented_image==0, kernel) & segmented_image

# 		boundary_mask = np.zeros([im.shape[0],im.shape[1],4])
# 		boundary_mask[:,:,-1] = segmented_boundaries

# 		plt.imshow(imarray)
# 		plt.imshow(boundary_mask)
# 		plt.show()

# 	return segmented_image


# def binary_mask_to_shapes(mask, im, cutoff=4, smooth=False):
# 	"""Initial segmentation step returns mask of 0s (background) and 1s (shapes).
# 	This function converts all groups of connected pixels (bigger than the cutoff size) 
# 	into Shape objects and stores them within a dictionary."""

# 	## np.argwhere(x==1) returns array of [row,column] 
# 	shape_pixels = np.argwhere(mask==1)
# 	shape_pixels_set = set([tuple(x) for x in shape_pixels])

# 	cells = dict()
# 	new_shapes_list = add_shapes_from_pixels(shape_pixels_set, cutoff)

# 	for s in new_shapes_list:
# 		cells[len(cells)+1] = Shape(s, im, smooth)

# 	return cells







if __name__ == "__main__":

	im = cv2.imread('cells.tif', cv2.IMREAD_UNCHANGED)
	imarray = np.array(im.reshape(im.shape), dtype=np.float32)

	## Blur the image to get rid of patchy artifacts
	img_blur = cv2.GaussianBlur(imarray,(3,3), sigmaX=0, sigmaY=0)

	cells = segmentation(dict(), img_blur, cutoff=10)

	print('No of cells: ', len(cells))

	fig, ax = plt.subplots()

	ax.imshow(imarray)

	cell_id_to_lines = {}

	for c in cells.keys():
		edges = cells[c].boundary
		cell_id_to_lines[c], = ax.plot(edges[:,1], edges[:,0], c='k')

	cellsplitter = CellSplitter(fig, ax, cells)
	
	plt.show()

	print('No of cells: ', len(cells))
	







