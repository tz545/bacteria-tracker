import numpy as np 
import matplotlib.pyplot as plt
import torch
import cv2
from collections import deque

from unet import UNet
from cells_manipulation import Shape, CellSplitter, mask_to_cells, manual_correction


def unet_segment_images(model_file, no_classes, image_file, image_number):

	model = UNet(1, no_classes)
	model.load_state_dict(torch.load(model_file))
	model.eval()

	im = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
	imarray = np.array(im.reshape(im.shape), dtype=np.float32)

	assert imarray.shape[0] % 2 == 0 and imarray.shape[1] % 2 == 0, "Input image dimensions should be even."
	quadrants = [imarray[:imarray.shape[0]//2, :imarray.shape[0]//2],\
				imarray[imarray.shape[0]//2:, :imarray.shape[0]//2],\
				imarray[:imarray.shape[0]//2, imarray.shape[0]//2:],\
				imarray[imarray.shape[0]//2:, imarray.shape[0]//2:]]

	for q in range(len(quadrants)):

		image = torch.from_numpy(quadrants[q])

		image = image.float()
		image = image/np.max(quadrants[q]) # normalize input values to between 0 and 1
		image = torch.unsqueeze(image, 0)
		image = torch.unsqueeze(image, 0)

		mask = torch.argmax(model(image), dim=1)
		mask = mask.squeeze().detach().numpy()

		cells = mask_to_cells(quadrants[q], mask)
		corrected_data = manual_correction(cells, quadrants[q])

		np.save('data/cell_{0}.npy'.format(image_number+q), corrected_data)


if __name__ == "__main__":

	unet_segment_images("models/low_contrast_19_epochs.pt", 4, 'cells_test.tif', 8)