import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler, Subset
from torchvision import datasets, transforms
import settings as sett



class DatasetLoader:
    def __init__(self, batch_size):
        self.__batch_size = batch_size
        self.__image_size = sett.image_size
        self.__small_dataset = sett.small_size
        self.__small_size = sett.small_size
        self.__train_path = sett.train_path
        self.__source_name = sett.source_name
        self.__target_name = sett.target_name

        self.train_data, self.val_data = self.load_training(self.__train_path, self.__source_name)
        self.__norm_mean, self.__norm_std = self.norm_std_mean()
        self.target_data = self.load_testing(self.__train_path, self.__target_name)

    def norm_std_mean(self):
        data_cumul = None
        for i, (data, label) in enumerate(self.train_data):
            if i == 0:
                data_cumul = data
            else:
                data_cumul = torch.cat((data_cumul, data), dim=0)
        return data_cumul.mean(), data_cumul.std()

    def load_training(self, root_path, dir):
        transform = transforms.Compose(
            [transforms.Resize([self.__image_size[0], self.__image_size[1]]),
             transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize(mean=[torch.tensor(0.7290)], std=[0.3436])
             ])
        data = datasets.ImageFolder(root=root_path + dir, transform=transform)

        ### SMALL DATASET FOR EXPERIMENTS ###
        if self.__small_dataset:
            small_ds_inds = np.random.choice(range(len(data)), size=int(len(data) * self.__small_size),
                                             replace=False)
            data = Subset(data, small_ds_inds)
        # if weighted:
        #     weights = make_weights_for_balanced_classes(data.imgs, len(data.classes))
        #     sampler = WeightedRandomSampler(weights, len(weights))
        #     train_loader = DataLoader(data, batch_size=batch_size, sampler=sampler, drop_last=True)
        # else:
        #     train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

        # np.random.seed(sett.seed)  # TODO: saving indices for test phase
        train_inds, val_inds = self.train_val_holdout_split(data, ratios=[0.8, 0.2])
        train_sampler = SubsetRandomSampler(train_inds)
        val_sampler = SubsetRandomSampler(val_inds)

        train_loader = DataLoader(data, batch_size=self.__batch_size, sampler=train_sampler, drop_last=True)
        val_loader = DataLoader(data, batch_size=self.__batch_size, sampler=val_sampler, drop_last=True)

        return train_loader, val_loader

    def load_testing(self, root_path, dir):
        transform = transforms.Compose(
            [transforms.Resize([self.__image_size[0], self.__image_size[1]]),
             transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize(mean=[self.__norm_mean], std=[self.__norm_std])])
        data = datasets.ImageFolder(root=root_path + dir, transform=transform)

        test_loader = DataLoader(data, batch_size=self.__batch_size, shuffle=True, drop_last=True)
        return test_loader

    def train_val_holdout_split(self, dataset, ratios=None):
        """Return indices for subsets of the dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset made with class which inherits `torch.utils.data.Dataset`
        ratios : list of floats
            List of [train, val] ratios respectively. Note, that sum of
            values must be equal to 1. (train + val = 1.0)
        """

        if ratios is None:
            ratios = [0.8, 0.2]
        assert np.allclose(ratios[0] + ratios[1], 1)
        train_ratio = ratios[0]
        val_ratio = ratios[1]

        df_size = len(dataset)
        train_inds = np.random.choice(range(df_size),
                                      size=int(df_size * train_ratio),
                                      replace=False)
        val_inds = list(set(range(df_size)) - set(train_inds))

        assert len(list(set(train_inds) - set(val_inds))) == len(train_inds)

        return train_inds, val_inds
