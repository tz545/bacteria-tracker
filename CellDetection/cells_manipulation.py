import numpy as np 
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from collections import deque
import matplotlib.pyplot as plt 

from boundary_edges import alpha_shape, stitch_boundaries

class Shape():

	def __init__(self, points):

		self.points = points # set of pixel indices (corresponding to original image) within Shape
		self._boundary = None
		self.center = np.mean(np.array(list(self.points)), axis=0)
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
		

	def to_dict(self):
		print(self.points)
		print(list(self.points))
		return {'points': list(self.points), 'boundary': self.boundary.tolist(), 'center': list(self.center), 'size': self.size}


class CellSplitter:
	"""Interactive interface for deleting and selecting cells on Matplotlib figure.
	Interactive controls:

	SINGLE RIGHT CLICK: remove cell
	LEFT CLICK and DRAG: draw region of cell"""


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
		self.cells[new_key] = Shape(new_shape)
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


def mask_to_cells(mask, min_cutoff=100, max_cutoff=10000, return_dict=False):
	"""This function converts all groups of connected pixels in a mask (between the cutoff sizes) 
	into Shape objects and stores them within a dictionary.
	mask: integer numpy array of same size as image, with value corresponding to the number of shapes present at pixel location"""

	new_shapes_list = []

	## BFS to identify neighbours  
	while np.sum(mask) > 0:

		# print(np.sum(mask))
		new_shape = set()
		cell_pixels = np.argwhere(mask==1) # we start searches on pixels with non-overlapping cells
		
		try:
			root_pixel = tuple(cell_pixels[0])
		except IndexError:
			print("", cell_pixels)
			print(np.histogram(mask.flatten(), bins=[-1.5, -0.5, 0.5, 1.5, 2.5])[0])
			break

		seen = np.zeros(mask.shape, dtype=int)

		BFS_queue = deque()
		BFS_queue.append(root_pixel)

		mask[cell_pixels[0,0], cell_pixels[0,1]] -= 1

		print_bool = False

		while len(BFS_queue) > 0:


			pixel = BFS_queue.popleft()
			new_shape.add(pixel)

			## search for 4-neighbours of the pixel
			row = pixel[0]
			col = pixel[1]

			neighboring_pixels = [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]
			for p in neighboring_pixels:
				if p[0] >= 0 and p[0] < mask.shape[0] and p[1] >= 0 and p[1] < mask.shape[1]:

					## only add neighbouring pixels if they contain an equal or greater number of cells
					if mask[p[0], p[1]] > mask[row, col] and seen[p[0], p[1]]==0:			
						mask[p[0], p[1]] -= 1
						BFS_queue.append(p)

					seen[p[0], p[1]] = 1		

		## we want to apply a cutoff to remove unconnected dots
		if len(new_shape) > min_cutoff and len(new_shape) < max_cutoff:
			new_shapes_list.append(new_shape)


	cells = {}

	for s in new_shapes_list:
		if return_dict:
			cells[len(cells)+1] = Shape(s).to_dict()
		else:
			cells[len(cells)+1] = Shape(s)

	return cells


def pixels_to_shapes(pixels, cutoff):
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

			new_pixel = BFS_queue.popleft()
			new_shape.add(new_pixel)

			## search for 4-neighbours of the pixel
			row = new_pixel[0]
			col = new_pixel[1]

			neighboring_pixels = [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]
			for p in neighboring_pixels:
				if p in pixels:
					pixels.remove(p)
					BFS_queue.append(p)

		## we want to apply a cutoff to remove unconnected dots
		if len(new_shape) > cutoff:
			new_shapes_list.append(new_shape)

	return new_shapes_list


def manual_correction(cells, pre_processed_image):
	"""Allows interactive user correction of segmentation.
	Interactive controls:

	SINGLE RIGHT CLICK: remove cell
	LEFT CLICK and DRAG: draw region of cell"""
	
	print('No of cells: ', len(cells))

	fig, ax = plt.subplots()
	ax.imshow(pre_processed_image)

	cell_id_to_lines = {}

	for c in cells.keys():
		edges = cells[c].boundary
		cell_id_to_lines[c], = ax.plot(edges[:,1], edges[:,0], c='k')

	cellsplitter = CellSplitter(fig, ax, pre_processed_image, cells, cell_id_to_lines)
	cellsplitter.connect()
	
	plt.show()

	cellsplitter.disconnect()

	mask = np.zeros(pre_processed_image.shape, dtype=int)


	for cell in cells.values():

		## convert indices list to matrix
		points_array = np.array(list(cell.points))
		row = points_array[:,0]
		col = points_array[:,1]

		mask[row,col] += 1

	return np.stack([pre_processed_image, mask])


def correct_masks(data_file):
	"""Allows visual inspection and correction of masks for quality control before training"""

	saved = np.load(data_file)
	image = saved[0]
	mask = saved[1]

	plt.imshow(mask, cmap='binary', alpha=0.3)
	plt.show()

	cells = mask_to_cells(image, mask)
	corrected_data = manual_correction(cells, image)

	np.save(data_file, corrected_data)


if __name__ == "__main__":
	
	correct_masks("data/cell_9.npy")