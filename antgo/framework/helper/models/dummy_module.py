import torch
from torch.utils.data import Dataset


class DummyModelWarp(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, image):
        # forward_dumpy is no redundant logic 
        return self.model.forward_dummy(image)

class UnlabeledDatasetWrapper(Dataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        images = self.dataset[index][self.key]
        return images

class LabeledDatasetWrapper(Dataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        images = self.dataset[index][self.key]
        return images, {}
