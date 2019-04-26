import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler, Subset
from torchvision import datasets, transforms
import settings as sett

def load_training(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([sett.image_size[0], sett.image_size[1]]),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.Normalize(mean=[torch.tensor(0.7290)], std=[0.3436])
         ])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)

    ### SMALL DATASET FOR EXPERIMENTS ###
    if sett.small_dataset:
        small_ds_inds = np.random.choice(range(len(data)), size=int(len(data)*sett.small_size),
                                            replace=False)
        data = Subset(data, small_ds_inds)
    # if weighted:
    #     weights = make_weights_for_balanced_classes(data.imgs, len(data.classes))
    #     sampler = WeightedRandomSampler(weights, len(weights))
    #     train_loader = DataLoader(data, batch_size=batch_size, sampler=sampler, drop_last=True)
    # else:
    #     train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    # np.random.seed(sett.seed)  # TODO: saving indices for test phase
    train_inds, val_inds = train_val_holdout_split(data, ratios=[0.8,0.2])
    train_sampler = SubsetRandomSampler(train_inds)
    val_sampler = SubsetRandomSampler(val_inds)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler, drop_last=True)

    return train_loader, val_loader

def load_testing(root_path, dir, batch_size, norm_mean, norm_std):
    transform = transforms.Compose(
        [transforms.Resize([sett.image_size[0], sett.image_size[1]]),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.Normalize(mean=[norm_mean], std=[norm_std])])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)

    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)
    return test_loader

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight  

def train_val_holdout_split(dataset, ratios=[0.8, 0.2]):
    """Return indices for subsets of the dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset made with class which inherits `torch.utils.data.Dataset`
    ratios : list of floats
        List of [train, val] ratios respectively. Note, that sum of 
        values must be equal to 1. (train + val = 1.0)
    """

    assert np.allclose(ratios[0] + ratios[1], 1)
    train_ratio = ratios[0]
    val_ratio = ratios[1]

    df_size = len(dataset)
    train_inds = np.random.choice(range(df_size), 
                                  size=int(df_size*train_ratio),
                                  replace=False)
    val_inds = list(set(range(df_size)) - set(train_inds))

    assert len(list(set(train_inds) - set(val_inds))) == len(train_inds)

    return train_inds, val_inds


train_loader, val_loader = load_training(sett.train_path, sett.source_name, sett.batch_size)
for i, (data, label) in enumerate(train_loader):
    if i == 0:
        data_cumul = data
    else:
        data_cumul = torch.cat((data_cumul, data), dim=0)
train_mean, train_std = data_cumul.mean(), data_cumul.std()

target_loader = load_testing(sett.train_path, sett.target_name, sett.batch_size, train_mean, train_std)

len_train_dataset = len(train_loader.dataset)
len_target_dataset = len(target_loader.dataset)
len_train_loader = len(train_loader)
len_target_loader = len(target_loader)


if __name__ == "__main__":
    print(train_loader.dataset.dataset.classes)
    print(val_loader.dataset.dataset.classes)
    print(target_loader.dataset.classes)

    # print(next(iter(train_loader)))
