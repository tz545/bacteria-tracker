import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from boundary_edges import alpha_shape, stitch_boundaries
from collections import deque


class Shape():


	def __init__(self, points, imshape):
		self.imshape = imshape # size of original image
		self.points = self.smooth_shape(points) # set of pixel indices (corresponding to original image) within Shape
		self.boundary = self.calc_boundary()
		self.size = len(self.points) # can consider automatic thresholding of sizes
		

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
			
		return np.array(boundary_polygon)

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
		dilation = binary_dilation(mask, kernel)
		erosion = binary_erosion(dilation, kernel)

		pixels = np.argwhere(erosion==1)
		pixels_set = set([tuple(x) for x in pixels])

		return pixels_set

	def delete_shape(self):
		pass


	def segment_shape(self):
		## apply watershed image segregation to split shape further 
		pass



def raw_image_to_binary_mask(image, segmentation, show_mask=False):
	"""Initial segmentation of raw image. Segmentation options are:
	'threshold', using 1.5x the median brightness value, or
	'kmeans', using k-means clustering with k=2"""

	im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
	imarray = np.array(im.reshape(im.shape), dtype=np.float32)

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

		boundary_mask = np.zeros([im.shape[0],im.shape[1],4])
		boundary_mask[:,:,-1] = segmented_boundaries

		plt.imshow(imarray)
		plt.imshow(boundary_mask)
		plt.show()

	return segmented_image



def binary_mask_to_shapes(mask, imshape, cutoff=4):
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
			cells[cell_no] = Shape(new_shape, imshape)

	return cells




class CellSplitter:
	def __init__(self, fig, axes, cells):
		self.fig = fig
		self.axes = axes
		self.cells = cells
		# self.xs = list(line.get_xdata())
		# self.ys = list(line.get_ydata())
		self.cid = fig.canvas.mpl_connect('button_press_event', self)

	def __call__(self, event):
		print('click', event)
		if event.inaxes!=self.axes: return



		if event.dblclick and event.button == 1:
			row = int(np.rint(event.ydata))
			col = int(np.rint(event.xdata))

			for c in cells.values():
				if (row, col) in c.points:
					ax.scatter(col, row)
					fig.canvas.draw()


		# self.xs.append(event.xdata)
		# self.ys.append(event.ydata)
		# self.line.set_data(self.xs, self.ys)
		# self.line.figure.canvas.draw()


if __name__ == "__main__":

	im = cv2.imread('cells.tif', cv2.IMREAD_UNCHANGED)
	imarray = np.array(im.reshape(im.shape), dtype=np.float32)

	mask = raw_image_to_binary_mask('cells.tif', segmentation='kmeans', show_mask=False)
	cells = binary_mask_to_shapes(mask, im.shape, cutoff=10)

	fig, ax = plt.subplots()

	ax.imshow(imarray)
	for c in cells.values():
		edges = c.boundary
		ax.plot(edges[:,1], edges[:,0], c='k')

	cellsplitter = CellSplitter(fig, ax, cells)
	
	plt.show()

	







