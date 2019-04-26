import torch
import torch.nn as nn
from coral import CORAL


def conv2d(in_channels, out_channels, padding, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ELU(alpha=0.7),
        # nn.Dropout2d(),
        nn.MaxPool2d(kernel_size=2)
    )


def dense(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_features=in_channels, out_features=out_channels),
        nn.BatchNorm1d(num_features=out_channels),
        nn.ELU(alpha=0.7),
        # nn.Dropout()
    )


def dense_softmax(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_features=in_channels, out_features=out_channels),
        nn.Softmax(dim=0)
    )


class ClassifierModel(nn.Module):
    def __init__(self, n_classes):
        super(ClassifierModel, self).__init__()

        self.layer0 = conv2d(1, 64, 0)
        self.layer1 = conv2d(64, 128, 0)
        self.layer2 = conv2d(128, 256, 0)
        self.layer3 = conv2d(256, 512, 0)
        self.layer4 = dense(2048, 1024)
        self.layer5 = dense_softmax(1024, n_classes)


    def forward(self, source, target=None):
        
        source = self.layer0(source)
        if target is not None:
            target = self.layer0(target)
            coral_loss = CORAL(source, target)
        else:
            coral_loss = None

        source = self.layer1(source)
        if target is not None:
            target = self.layer1(target)
            coral_loss = torch.cat((coral_loss, CORAL(source, target)))
        
        source = self.layer2(source)
        if target is not None:
            target = self.layer2(target)
            coral_loss = torch.cat((coral_loss, CORAL(source, target)))

        source = self.layer3(source)
        if target is not None:
            target = self.layer3(target)
            coral_loss = torch.cat((coral_loss, CORAL(source, target)))

        source = source.view(source.size(0), -1)
        if target is not None:
            target = target.view(source.size(0), -1)
        
        source = self.layer4(source)
        if target is not None:
            target = self.layer4(target)
            coral_loss = torch.cat((coral_loss, CORAL(source, target)))
        
        source = self.layer5(source)
        if target is not None:
            target = self.layer5(target)
            coral_loss = torch.cat((coral_loss, CORAL(source, target)))
        return source, coral_loss

