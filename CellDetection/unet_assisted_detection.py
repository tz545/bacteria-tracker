import numpy as np 
import matplotlib.pyplot as plt
import torch
import cv2
from collections import deque
# import wandb

from unet import UNet
from cells_manipulation import CellSplitter, mask_to_cells, manual_correction


def unet_segment_images(model_file, no_classes, image_file, image_number=0, save=False):

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
			np.save('data/cell_{0}.npy'.format(image_number+q), corrected_data)


if __name__ == "__main__":

	# wandb.login()
	# api = wandb.Api()

	# sweep = api.sweep("tz545/UNet-cell-detection/s6dl6p8k")
	# runs = sorted(sweep.runs,key=lambda run: run.summary.get("val_loss", 0), reverse=False)
	# val_loss = runs[0].summary.get("val_loss", 0)
	# print(f"Best run {runs[0].name} with {val_loss} validation loss")

	# print(runs[0].file("tz545/UNet-cell-detection/wrmn6w8j/.h5"))

	# runs[0].file("tz545/UNet-cell-detection/435xuaxr.h5").download(replace=True)
	# print("Best model saved to model-best.h5")

	unet_segment_images("models/low_contrast_expanded_dataset_sweep/50_epochs_lr_0.0005_m_0.1_best.pt", 4, 'cells_images/PA_vipA_mnG_30x30_32x32_35nN_100uNs_2s_1_GFP-1.tif')