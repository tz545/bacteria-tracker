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
		self.press = False
		self.addpoints = []
		self.radius = 5

	def connect(self):
		self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
		self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
		self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)


	def on_press(self, event):
		"""Delete cell or add new cell"""

		if event.inaxes != self.axes: return

		row = int(np.rint(event.ydata))
		col = int(np.rint(event.xdata))

		## right click on mouse to remove falsely identified cells
		if not event.dblclick and event.button == 3:
			to_delete = []
			
			for c in self.cells.keys():
				if (row, col) in self.cells[c].points:
					self.cell_id_to_lines[c].remove()
					to_delete.append(c)

			for c_d in to_delete:
				self.cells.pop(c_d)
				self.cell_id_to_lines.pop(c_d)


			self.fig.canvas.draw()

			return

		## left click and drag on mouse to draw new cell
		elif not event.dblclick and event.button == 1:
			self.press = True

	def on_motion(self, event):
		"""Keep track of all pixels passed over by mouse."""

		if not self.press or event.inaxes != self.axes:
			return

		row = int(np.rint(event.ydata))
		col = int(np.rint(event.xdata))

		self.addpoints.append((row, col))


	def on_release(self, event):
		"""Add cell and clear button press information"""

		if not self.press:
			return

		if len(self.addpoints) < 4:
			self.press = False
			self.addpoints = []
			return 

		new_shape = set()

		for point in self.addpoints:
			for i in range(-self.radius, self.radius+1):
				for j in range(-self.radius, self.radius+1):
					if abs(i) + abs(j) <= self.radius:
						if point[0]+i >= 0 and point[0]+i <= self.image.shape[0]-1 and point[1] + j >= 0 and point[1] + j <= self.image.shape[1]-1:
							
							new_shape.add((point[0]+i, point[1]+j))

		new_key = max(self.cells.keys())+1
		self.cells[new_key] = Shape(new_shape, self.image, smooth=False)
		edges = self.cells[new_key].boundary
		self.cell_id_to_lines[new_key], = self.axes.plot(edges[:,1], edges[:,0], c='k')
		
		self.press = False
		self.addpoints = []
		self.fig.canvas.draw()


	def disconnect(self):
		"""Disconnect all callbacks"""

		self.fig.canvas.mpl_disconnect(self.cidpress)
		self.fig.canvas.mpl_disconnect(self.cidrelease)
		self.fig.canvas.mpl_disconnect(self.cidmotion)



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


def segmentation_to_shapes(cells, image, segmentation_function, *threshold, points=None, cutoff=4, smooth=True):
	"""Either an identified region needs to be split into multiple cells
	or an unselected region contains cells that need to be segmented
	image: 2D numpy array of pixel values
	points: set of (tuple) pixel coordinates that will undergo another round of segmentation"""

	segmented_points = segmentation_function(image, *threshold, points)

	shape_pixels_set = set([tuple(x) for x in segmented_points])

	new_shapes_list = add_shapes_from_pixels(shape_pixels_set, cutoff)

	for s in new_shapes_list:
		if len(cells) == 0:
			cells[len(cells)+1] = Shape(s, image, smooth)
		else:
			cells[max(cells.keys())+1] = Shape(s, image, smooth)



if __name__ == "__main__":

	im = cv2.imread('cells.tif', cv2.IMREAD_UNCHANGED)
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

		cells_single_frame = {}
		segmentation_to_shapes(cells_single_frame, img_blur.astype(np.float32), threshold_segmentation, 1.5, cutoff=100, smooth=False)
		
		print('No of cells: ', len(cells_single_frame))

		fig, ax = plt.subplots()
		ax.imshow(quadrants[q])

		cell_id_to_lines = {}

		for c in cells_single_frame.keys():
			edges = cells_single_frame[c].boundary
			cell_id_to_lines[c], = ax.plot(edges[:,1], edges[:,0], c='k')

		cellsplitter = CellSplitter(fig, ax, quadrants[q], cells_single_frame, cell_id_to_lines)
		cellsplitter.connect()
		
		plt.show()

		cellsplitter.disconnect()

		mask = np.zeros(quadrants[q].shape, dtype=int)


		for cell in cells_single_frame.values():

			## convert indices list to matrix
			points_array = np.array(list(cell.points))
			row = points_array[:,0]
			col = points_array[:,1]

			mask[row,col] += 1


		print(np.max(mask))

		np.save('cell_{0}.npy'.format(q), np.stack([quadrants[q], mask]))



	







