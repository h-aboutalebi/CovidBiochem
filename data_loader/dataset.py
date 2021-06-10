from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import math
import torch


class TCNDataset(Dataset):

    def __init__(self, positive_samples, negative_samples):
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

    def __len__(self):
        return self.positive_samples.shape[0] + self.negative_samples.shape[0]

    def __getitem__(self, idx):
        if (idx % 2 == 0):
            label = 0
            x = self.negative_samples[math.floor(idx / 2)]
        else:
            label = 1
            x = self.positive_samples[math.floor(idx / 2)]
        return (x, label)

    # def make_weights_for_balanced_classes(self):
    #     count = [0,0]
    #     for i in range(self.positive_samples.shape[0]):
    #         count[1] += 1
    #     for i in range(self.negative_samples.shape[0]):
    #         count[0] += 1
    #     weight_per_class = [0,0]
    #     N = float(sum(count))
    #     for i in range(2):
    #         weight_per_class[i] = 1/float(count[i])
    #     weight = [0] * self.__len__()
    #     for i in range(self.__len__()):
    #         _,label=self.__getitem__(i)
    #         weight[i] = weight_per_class[label]
    #     return weight
