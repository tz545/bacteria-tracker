import numpy as np 
import matplotlib.pyplot as plt
import cv2
import torch
import os, sys, pathlib

current_folder = pathlib.Path(__file__).parent.resolve()
src_folder = str(current_folder.parent.absolute())
if src_folder not in sys.path:
	sys.path.insert(0, src_folder)

from utils import Shape, mask_to_cells, threshold_segmentation, kmeans_segmentation, segmentation_to_cells
from models.unet import UNet

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


def threshold_segment_images(image_file, image_number, save_loc):
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
		segmentation_to_cells(img_blur, cells, threshold_segmentation, 1.5, cutoff=100)
		image_with_mask = manual_correction(cells, quadrants[q])

		np.save(os.path.join(save_loc, 'cell_{0}.npy'.format(image_number+q)), image_with_mask)


def unet_segment_images(model_file, no_classes, image_file, save_loc, image_number=0, save=False):

	model = UNet(1, no_classes)
	model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
	model.eval()

	im = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
	imarray = np.array(im.reshape(im.shape), dtype=np.float32)

	assert imarray.shape[0] % 2 == 0 and imarray.shape[1] % 2 == 0, "Input image dimensions should be even."
	quadrants = [imarray[:imarray.shape[0]//2, :imarray.shape[0]//2],\
				imarray[imarray.shape[0]//2:, :imarray.shape[0]//2],\
				imarray[:imarray.shape[0]//2, imarray.shape[0]//2:],\
				imarray[imarray.shape[0]//2:, imarray.shape[0]//2:]]

	for q in range(1):

		image = torch.from_numpy(quadrants[q])

		image = image.float()
		# image = image - torch.min(image).item()
		image = image/torch.max(image).item()
		image = torch.unsqueeze(image, 0)
		image = torch.unsqueeze(image, 0)

		mask = torch.argmax(model(image), dim=1)
		mask = mask.squeeze().detach().numpy()

		cells = mask_to_cells(mask)
		corrected_data = manual_correction(cells, quadrants[q])

		if save:
			np.save(os.path.join(save_loc, 'cell_{0}.npy'.format(image_number+q)), corrected_data)


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

	project_folder = pathlib.Path(src_folder).parent.absolute()
	data_folder = os.path.join(project_folder, 'data')
	model_folder = os.path.join(project_folder, 'models')

	# ## threshold segement test image
	# threshold_segment_images(os.path.join(data_folder, 'raw', 'PA_vipA_mnG_30x30_32x32_35nN_100uNs_2s_1_GFP-1.tif'), 0, os.path.join(data_folder, 'training'))

	## UNet segment test image
	# unet_segment_images(os.path.join(model_folder, "unet_02.pt"), 4, os.path.join(data_folder, 'raw', 'PA_vipA_mnG_30x30_32x32_35nN_100uNs_2s_1_GFP-1.tif'), os.path.join(data_folder, 'training'))

	correct_masks(os.path.join(data_folder, "training", "cell_7.npy"))