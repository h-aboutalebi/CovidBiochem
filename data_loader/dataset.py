from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch

class TCNDataset(Dataset):

    def __init__(self, positive_samples,negative_samples):
        self.positive_samples=positive_samples
        self.negative_samples=negative_samples

    def __len__(self):
        return self.positive_samples.shape[0]+self.negative_samples.shape[0]



