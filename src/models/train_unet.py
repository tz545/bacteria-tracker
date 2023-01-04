import os
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from models.unet import UNet
import matplotlib.pyplot as plt


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
			low_norm = low_norm - torch.min(low_norm).item()
			low_norm = low_norm/torch.max(low_norm).item()

			high_norm = TF.normalize(image, -1, 2)
			high_norm = high_norm - torch.min(high_norm).item()
			high_norm = high_norm/torch.max(high_norm).item()

			self.samples.append({'image': low_norm, 'mask': mask})
			self.samples.append({'image': high_norm, 'mask': mask})

			low_brightness = TF.adjust_brightness(image, 0.5) 
			low_brightness = low_brightness - torch.min(low_brightness).item()
			low_brightness = low_brightness/torch.max(low_brightness).item()

			med_brightness = TF.adjust_brightness(image, 0.75)
			med_brightness = med_brightness - torch.min(med_brightness).item()
			med_brightness = med_brightness/torch.max(med_brightness).item()

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
			low_contrast1 = low_contrast1 - torch.min(low_contrast1).item()
			low_contrast1 = low_contrast1/torch.max(low_contrast1).item()
			low_contrast1_hflip = TF.hflip(low_contrast1)
			low_contrast1_vflip = TF.vflip(low_contrast1)

			self.samples.append({'image': low_contrast1, 'mask': mask})
			self.samples.append({'image': low_contrast1_hflip, 'mask': horizontal_flip_mask})
			self.samples.append({'image': low_contrast1_vflip, 'mask': vertical_flip_mask})

			low_contrast2 = TF.normalize(image, -5, 40)
			low_contrast2 = low_contrast2 - torch.min(low_contrast2).item()
			low_contrast2 = low_contrast2/torch.max(low_contrast2).item()
			low_contrast2_hflip = TF.hflip(low_contrast2)
			low_contrast2_vflip = TF.vflip(low_contrast2)

			self.samples.append({'image': low_contrast2, 'mask': mask})
			self.samples.append({'image': low_contrast2_hflip, 'mask': horizontal_flip_mask})
			self.samples.append({'image': low_contrast2_vflip, 'mask': vertical_flip_mask})

			low_contrast3 = TF.normalize(image, -30, 40)
			low_contrast3 = low_contrast3 - torch.min(low_contrast3).item()
			low_contrast3 = low_contrast3/torch.max(low_contrast3).item()
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

	no_trainng_samples = int(0.8*len(dataset.samples))
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


	torch.save(model.state_dict(), "models/low_contrast_expanded_dataset/{0}_epochs.pt".format(epochs))

	return train_loss, val_loss




if __name__ == "__main__":

	dataset = CellsDataset('data')
	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	for i, batch in enumerate(loader):
		image = batch['image']
		image = image.squeeze()
		mask = batch['mask']
		mask = mask.squeeze()

		plt.imshow(image)
		plt.colorbar()
		plt.imshow(mask, cmap="Greys", alpha=0.3)
		plt.show()

	