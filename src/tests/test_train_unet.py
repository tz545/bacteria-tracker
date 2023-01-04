import numpy as np 
import torch
import pytest
import os
from models.train_unet import CellsDataset
from torch.utils.data import DataLoader


@pytest.fixture
def rootdir():
	return os.path.dirname(os.path.abspath(__file__))

## schema tests: check shapes, dtypes, min/max of image and mask tensors

def test_dataset_output_shapes(rootdir):
	testset = CellsDataset(os.path.join(rootdir, "test_dataset"))
	loader = DataLoader(testset, batch_size=1, shuffle=False)
	for i, batch in enumerate(loader):
		assert batch['image'].shape == torch.Size([1, 1, 1024, 1024])
		assert batch['mask'].shape == torch.Size([1, 1024, 1024])
	loader_batch = DataLoader(testset, batch_size=3, shuffle=False)
	batch = next(iter(loader_batch))
	assert batch['image'].shape == torch.Size([3, 1, 1024, 1024])
	assert batch['mask'].shape == torch.Size([3, 1024, 1024])


def test_dataset_output_dtypes(rootdir):
	testset = CellsDataset(os.path.join(rootdir, "test_dataset"))
	loader = DataLoader(testset, batch_size=1, shuffle=False)
	for i, batch in enumerate(loader):
		assert batch['image'].type() == 'torch.FloatTensor'
		assert batch['mask'].type() == 'torch.LongTensor'
		

def test_dataset_output_min_max(rootdir):
	testset = CellsDataset(os.path.join(rootdir, "test_dataset"))
	loader = DataLoader(testset, batch_size=1, shuffle=False)
	for i, batch in enumerate(loader):
		assert torch.max(batch['image']).item() <= 1
		assert torch.min(batch['image']).item() >= 0
		assert torch.min(batch['mask']).item() == 0
		assert torch.max(batch['mask']).item() <= 3

