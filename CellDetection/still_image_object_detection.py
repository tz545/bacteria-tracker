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
		if self.size < 10:
			print("Small size detected! ", self.size)

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
	def __init__(self, fig, axes, image, cells, cell_id_to_lines):
		self.fig = fig
		self.axes = axes
		self.cells = cells 
		self.image = image
		self.cell_id_to_lines = cell_id_to_lines
		self.cid = fig.canvas.mpl_connect('button_press_event', self)

	def __call__(self, event):
		print('click', event)
		if event.inaxes != self.axes: return

		to_delete = []
		## double click on mouse to remove falsely identified cells
		if event.dblclick and event.button == 1:
			row = int(np.rint(event.ydata))
			col = int(np.rint(event.xdata))

			for c in self.cells.keys():
				if (row, col) in self.cells[c].points:
					self.cell_id_to_lines[c].remove()
					to_delete.append(c)
		
		to_segment = []
		## single right click on mouse on identified regions to segment further
		if not event.dblclick and event.button == 3:
			row = int(np.rint(event.ydata))
			col = int(np.rint(event.xdata))

			for c in self.cells.keys():
				if (row, col) in self.cells[c].points:
					self.cell_id_to_lines[c].remove()
					to_segment.append(c)

		for c_d in to_delete:
			self.cells.pop(c)
			self.cell_id_to_lines.pop(c)

		for c_s in to_segment:
			cell_c = self.cells.pop(c_s)
			self.cell_id_to_lines.pop(c_s)
			segmentation_to_shapes(self.cells, self.image, points=cell_c.points, cutoff=10)
			for cell in self.cells.keys():
				if cell not in self.cell_id_to_lines.keys():
					edges = self.cells[cell].boundary
					self.cell_id_to_lines[cell], = self.axes.plot(edges[:,1], edges[:,0], c='k')


		self.fig.canvas.draw()



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


def threshold_segmentation(image, points=None):

	if points == None:
		binary_mask = np.sign(image - 1.25*np.median(image)).astype(int)
		return np.argwhere(binary_mask==1)
		
	else:
		points_array = np.array(list(points))
		im_vals = image[points_array[:,0], points_array[:,1]]
		binary_mask = np.sign(im_vals - 0.9*np.median(im_vals)).astype(int)	
		return points_array[binary_mask==1]



def add_shapes_from_pixels(pixels, cutoff):
	"""This function converts all groups of connected pixels (bigger than the cutoff size) 
 	into Shape objects and stores them within a dictionary.
	pixels: set of (tuple) pixel coordinates"""

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


def segmentation_to_shapes(cells, image, points=None, cutoff=4, smooth=True):
	"""Either an identified region needs to be split into multiple cells
	or an unselected region contains cells that need to be segmented
	image: 2D numpy array of pixel values
	points: set of (tuple) pixel coordinates that will undergo another round of kmeans segmentation"""

	segmented_points = threshold_segmentation(image, points)

	shape_pixels_set = set([tuple(x) for x in segmented_points])

	new_shapes_list = add_shapes_from_pixels(shape_pixels_set, cutoff)

	for s in new_shapes_list:
		if len(cells) == 0:
			cells[len(cells)+1] = Shape(s, im, smooth)
		else:
			cells[max(cells.keys())+1] = Shape(s, im, smooth)



if __name__ == "__main__":

	im = cv2.imread('cells.tif', cv2.IMREAD_UNCHANGED)
	imarray = np.array(im.reshape(im.shape), dtype=np.float32)

	## Blur the image to get rid of patchy artifacts
	img_blur = cv2.GaussianBlur(imarray,(3,3), sigmaX=0, sigmaY=0)

	cells_frame1 = dict()
	segmentation_to_shapes(cells_frame1, img_blur, cutoff=10, smooth=False)

	print('No of cells: ', len(cells_frame1))

	fig, ax = plt.subplots()
	ax.imshow(imarray)

	cell_id_to_lines = {}

	for c in cells_frame1.keys():
		edges = cells_frame1[c].boundary
		cell_id_to_lines[c], = ax.plot(edges[:,1], edges[:,0], c='k')

	cellsplitter = CellSplitter(fig, ax, imarray, cells_frame1, cell_id_to_lines)
	
	plt.show()

	print('No of cells: ', len(cells_frame1))
	







