import cv2
import os
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from unet import UNet
import matplotlib.pyplot as plt
import wandb


class CellsDataset(Dataset):

	def __init__(self, datadir):
		self.datadir = datadir
		self.samples = []
		self._init_dataset()

	def _init_dataset(self):

		for cellfile in os.listdir(self.datadir):

			data = np.load(os.path.join(self.datadir, cellfile))
			image = torch.from_numpy(data[0])
			mask = torch.from_numpy(data[1])

			image = image.float()
			image = image/np.max(data[0]) # normalize input values to between 0 and 1
			image = torch.unsqueeze(image, 0)

			mask = mask.long()

			self.samples.append({'image': image, 'mask': mask})

			## apply transforms and append

			low_norm = TF.normalize(image, 0.0, 2)
			high_norm = TF.normalize(image, -1, 2)

			self.samples.append({'image': low_norm, 'mask': mask})
			self.samples.append({'image': high_norm, 'mask': mask})

			low_brightness = TF.adjust_brightness(image, 0.5) 
			med_brightness = TF.adjust_brightness(image, 0.75)

			self.samples.append({'image': low_brightness, 'mask': mask})
			self.samples.append({'image': med_brightness, 'mask': mask})

			horizontal_flip = TF.hflip(image)
			horizontal_flip_mask = TF.hflip(mask)

			vertical_flip = TF.vflip(image)
			vertical_flip_mask = TF.vflip(mask)

			self.samples.append({'image': horizontal_flip, 'mask': horizontal_flip_mask})
			self.samples.append({'image': vertical_flip, 'mask': vertical_flip_mask})

			## construct a set of lower-contrast data

			low_contrast1 = TF.normalize(image, -10, 20)
			low_contrast1_hflip = TF.hflip(low_contrast1)
			low_contrast1_vflip = TF.vflip(low_contrast1)

			self.samples.append({'image': low_contrast1, 'mask': mask})
			self.samples.append({'image': low_contrast1_hflip, 'mask': horizontal_flip_mask})
			self.samples.append({'image': low_contrast1_vflip, 'mask': vertical_flip_mask})

			low_contrast2 = TF.normalize(image, -5, 40)
			low_contrast2_hflip = TF.hflip(low_contrast2)
			low_contrast2_vflip = TF.vflip(low_contrast2)

			self.samples.append({'image': low_contrast2, 'mask': mask})
			self.samples.append({'image': low_contrast2_hflip, 'mask': horizontal_flip_mask})
			self.samples.append({'image': low_contrast2_vflip, 'mask': vertical_flip_mask})

			low_contrast3 = TF.normalize(image, -30, 40)
			low_contrast3_hflip = TF.hflip(low_contrast3)
			low_contrast3_vflip = TF.vflip(low_contrast3)

			self.samples.append({'image': low_contrast3, 'mask': mask})
			self.samples.append({'image': low_contrast3_hflip, 'mask': horizontal_flip_mask})
			self.samples.append({'image': low_contrast3_vflip, 'mask': vertical_flip_mask})


	def __len__(self):
		return len(self.samples)


	def __getitem__(self, idx):
		return self.samples[idx]




def train_net(dataset, batch_size, epochs, learning_rate=1e-5):

	no_trainng_samples = int(0.9*len(dataset.samples))
	no_val_samples = len(dataset.samples) - no_trainng_samples

	trainset, valset = random_split(dataset, [no_trainng_samples, no_val_samples])

	train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

	model = UNet(1, 4)
	optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)

	criterion = nn.CrossEntropyLoss()

	train_loss = []
	val_loss = []

	for e in range(1, epochs+1):

		total_train_loss = 0
		total_val_loss = 0

		for i, batch in enumerate(train_loader):

			optimizer.zero_grad()

			output = model(batch['image'])
			loss = criterion(output, batch['mask'])

			wandb.log({"loss": loss})

			loss.backward()
			optimizer.step()

			total_train_loss += loss.item()/batch_size

		print("Total training loss: ", total_train_loss/(i+1))

		with torch.no_grad(): 

			for i, batch in enumerate(val_loader):

				output = model(batch['image'])
				loss = criterion(output, batch['mask'])

				total_val_loss += loss.item()/batch_size

				val_loss.append(loss.item()/batch_size)

			print("Total validation loss: ", total_val_loss/(i+1))


	torch.save(model.state_dict(), "models/low_contrast_{0}_epochs.pt".format(epochs))

	return train_loss, val_loss




if __name__ == "__main__":

	wandb.init(project="UNet-cell-detection")

	wandb.config = {
		"learning_rate": 1e-5,
		"epochs": 50,
		"batch_size": 128
	}

	test = True

	if test:

		model = UNet(1, 4)
		model.load_state_dict(torch.load("models/low_contrast_19_epochs.pt"))
		model.eval()

		im = cv2.imread('cells_test.tif', cv2.IMREAD_UNCHANGED)
		imarray = np.array(im.reshape(im.shape), dtype=np.float32)

		assert imarray.shape[0] % 2 == 0 and imarray.shape[1] % 2 == 0, "Input image dimensions should be even."
		quadrants = [imarray[:imarray.shape[0]//2, :imarray.shape[0]//2],\
					imarray[imarray.shape[0]//2:, :imarray.shape[0]//2],\
					imarray[:imarray.shape[0]//2, imarray.shape[0]//2:],\
					imarray[imarray.shape[0]//2:, imarray.shape[0]//2:]]

		for q in range(len(quadrants)):
			# plt.imshow(quadrants[q])
			img_blur = cv2.GaussianBlur(quadrants[q],(3,3), sigmaX=10, sigmaY=10)

			image = torch.from_numpy(img_blur)

			image = image.float()
			image = image/np.max(img_blur) # normalize input values to between 0 and 1
			image = torch.unsqueeze(image, 0)
			image = torch.unsqueeze(image, 0)

			mask = torch.argmax(model(image), dim=1)
			print(mask.shape)

			plt.imshow(mask.squeeze().detach().numpy(), cmap='binary', alpha=0.3)
			plt.show()

	else:

		dataset = CellsDataset('data')
		train_net(dataset, batch_size=1, epochs=19)