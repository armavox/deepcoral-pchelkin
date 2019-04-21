import torch
import torch.nn as nn


def conv2d(in_channels, out_channels, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding),
        nn.ELU(alpha=0.7, inplace=True),
        nn.Dropout2d(inplace=True),
        nn.MaxPool2d(kernel_size=2)
    )


def dense(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_features=in_channels, out_features=out_channels),
        nn.ELU(alpha=0.7, inplace=True),
        nn.Dropout()
    )


def dense_softmax(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_features=in_channels, out_features=out_channels),
        nn.Softmax(dim=0)
    )


class ClassifierModel(nn.Module):
    def __init__(self, n_classes):
        super(ClassifierModel, self).__init__()

        self.layer1 = conv2d(5, 16, 0)
        self.layer2 = conv2d(16, 32, 19)
        self.layer3 = conv2d(32, 64, 19)
        self.layer4 = dense(78400, 64)
        self.layer5 = dense_softmax(64, n_classes)

    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out).view(-1)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
