import torch
import torch.nn as nn
from torch.autograd import Function, Variable


'''
MODELS
'''


# def CORAL(source, target):
#     d = source.data.shape[1]

#     # source covariance
#     xm = torch.mean(source, 0, keepdim=True) - source
#     xc = xm.t() @ xm

#     # target covariance
#     xmt = torch.mean(target, 0, keepdim=True) - target
#     xct = xmt.t() @ xmt

#     # frobenius norm between source and target
#     loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
#     loss = loss/(4*d*d)
#     print('CORAL_LOSS', loss)
#     return loss
def CORAL(source, target):
    d = source.data.shape[0]  # Batch size
    assert source[0].shape == target[0].shape, f'src.sh is {source.shape}; tg.sh is {target.shape}!'
    # Source covariance
    source = source - torch.mean(source, 1, keepdim=True)
    source_cov = torch.matmul(torch.transpose(source, -2,-1), source)
    # Target covariance
    target =  target - torch.mean(target, 1, keepdim=True)
    target_cov = torch.matmul(torch.transpose(target, -2, -1), target)

    loss = torch.norm((source_cov - target_cov), dim=(-2,-1))
    loss = loss/(4*d**2)
    loss = torch.mean(loss).unsqueeze(-1)
    return loss

class DeepCORAL(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepCORAL, self).__init__()
        self.sharedNet = AlexNet()
        self.fc = nn.Linear(4096, num_classes)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.sharedNet(source)
        
        coral_loss = CORAL(source, target)

        source = self.fc(source)

        target = self.sharedNet(target)
        target = self.fc(target)
        
        return source, target


class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2304, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, source, target=None):
        source = self.features(source)
        if target is not None:
            target = self.features(target)

        source = source.view(source.size(0), -1)
        if target is not None:
            target = target.view(target.size(0), -1)

        source = self.classifier(source)
        if target is not None:
            target = self.classifier(target)
            coral_loss = CORAL(source, target)

        source = self.fc(source)
        if target is not None:
            return source, coral_loss
        else:
            return source, None
