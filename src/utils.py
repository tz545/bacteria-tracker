import numpy as np 
from scipy.spatial import Delaunay
from collections import deque
import cv2

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
		return {'points': list(self.points), 'boundary': self.boundary.tolist(), 'center': list(self.center), 'size': self.size}


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


def segmentation_to_cells(image, cells, segmentation_function, *threshold, points=None, cutoff=4, return_dict=False):
	"""Either an identified region needs to be split into multiple cells
	or an unselected region contains cells that need to be segmented
	image: 2D numpy array of pixel values
	points: set of (tuple) pixel coordinates that will undergo another round of segmentation"""

	segmented_points = segmentation_function(image, *threshold, points)
	shape_pixels_set = set([tuple(x) for x in segmented_points])
	new_shapes_list = pixels_to_shapes(shape_pixels_set, cutoff)

	for s in new_shapes_list:
		if len(cells) == 0:
			if return_dict:
				cells[len(cells)+1] = Shape(s).to_dict()
			else:
				cells[len(cells)+1] = Shape(s)
		else:
			if return_dict:
				cells[max(cells.keys())+1] = Shape(s).to_dict()
			else:
				cells[max(cells.keys())+1] = Shape(s)

	return cells


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


## code for computing concave hull of a set of points
## taken from https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
def alpha_shape(points, alpha, only_outer=True):
	"""
	Compute the alpha shape (concave hull) of a set of points.
	:param points: np.array of shape (n,2) points.
	:param alpha: alpha value.
	:param only_outer: boolean value to specify if we keep only the outer border
	or also inner edges.
	:return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
	the indices in the points array.
	"""
	assert points.shape[0] > 3, "Need at least four points"

	def add_edge(edges, i, j):
		"""
		Add an edge between the i-th and j-th points,
		if not in the list already
		"""
		if (i, j) in edges or (j, i) in edges:
			# already added
			assert (j, i) in edges, "Can't go twice over same directed edge right?"
			if only_outer:
				# if both neighboring triangles are in shape, it's not a boundary edge
				edges.remove((j, i))
			return
		edges.add((i, j))

	tri = Delaunay(points)
	edges = set()
	# Loop over triangles:
	# ia, ib, ic = indices of corner points of the triangle
	for ia, ib, ic in tri.vertices:
		pa = points[ia]
		pb = points[ib]
		pc = points[ic]
		# Computing radius of triangle circumcircle
		# www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
		a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
		b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
		c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
		s = (a + b + c) / 2.0
		area = np.sqrt(s * (s - a) * (s - b) * (s - c))
		circum_r = a * b * c / (4.0 * area)
		if circum_r < alpha:
			add_edge(edges, ia, ib)
			add_edge(edges, ib, ic)
			add_edge(edges, ic, ia)
	return edges


def find_edges_with(i, edge_set):
	i_first = [j for (x,j) in edge_set if x==i]
	i_second = [j for (j,x) in edge_set if x==i]
	return i_first,i_second

def stitch_boundaries(edges):
	edge_set = edges.copy()
	boundary_lst = []
	while len(edge_set) > 0:
		boundary = []
		edge0 = edge_set.pop()
		boundary.append(edge0)
		last_edge = edge0
		while len(edge_set) > 0:
			i,j = last_edge
			j_first, j_second = find_edges_with(j, edge_set)
			if j_first:
				edge_set.remove((j, j_first[0]))
				edge_with_j = (j, j_first[0])
				boundary.append(edge_with_j)
				last_edge = edge_with_j
			elif j_second:
				edge_set.remove((j_second[0], j))
				edge_with_j = (j, j_second[0])  # flip edge rep
				boundary.append(edge_with_j)
				last_edge = edge_with_j

			if edge0[0] == last_edge[1]:
				break

		boundary_lst.append(boundary)
	return boundary_lst