import os
import numpy as np
import torch
import argparse
import torch.nn.functional as F
import torchvision.models as tvmodels
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import data_loader
import models_upd as models
import settings
import utils
from classifier import ClassifierModel
from original_model import AlexNet
    # LEARNING_RATE = settings.lr / math.pow((1 + 10 * (epoch - 1) / settings.epochs), 0.75)

def train(epoch, model, data_loader, optimizer, loss_func, device='cpu', deepcoral=False):
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print("""\n========== EPOCH {} of {} ===========
learning rate: {:.8f}""".format(epoch, settings.epochs, learning_rate) )
    
    if deepcoral:
        model.train()
        iter_source = iter(data_loader.train_loader)
        iter_target = iter(data_loader.target_loader)
        num_iter = data_loader.len_train_loader
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, label_target = iter_target.next()
            if i % data_loader.len_target_loader == 0:
                iter_target = iter(data_loader.target_loader)

            data_source, label_source = data_source.to(device), label_source.to(device)
            data_target, label_target = data_target.to(device), label_target.to(device)

            label_source_pred, loss_coral = model(data_source, data_target)
            loss_cls = loss_func(label_source_pred, label_source)
            gamma = 2  / (1 + np.exp(-10 * (epoch) / settings.epochs)) - 1
            loss_coral = torch.mean(loss_coral)
            loss = loss_cls + gamma * loss_coral
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % settings.log_interval == 0 or i == num_iter - 1:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\ttotal_Loss: {:.8f} cls_Loss: {:.8f} coral_Loss: {:.8f}'.format(
                    epoch, i * len(data_source), data_loader.len_train_dataset,
                    100. * i / data_loader.len_train_loader, loss.item(), loss_cls.item(), loss_coral.item()))
    else:
        model.train()
        iter_source = iter(data_loader.train_loader)
        num_iter = data_loader.len_train_loader
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_source, label_source = data_source.to(device), label_source.to(device)

            label_source_pred, _ = model(data_source)
            loss = loss_func(label_source_pred, label_source)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % settings.log_interval == 0 or i == num_iter-1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data_source), data_loader.len_train_dataset,
                    100. * i / data_loader.len_train_loader, 
                    loss.data))
    epoch_train.append(epoch)
    loss_train.append(loss)

def test(epoch, model, dataset_loader, loss_func, mode="Val", device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    pred_list = np.array([])
    target_list = np.array([])
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = data.to(device), target.to(device)
            if settings.deepcoral:
                output, _ = model(data)
            else:
                output, _ = model(data)
            test_loss += loss_func(output, target)#, reduction='sum')
            pred = output.data.max(1)[1] # get the index of the max log-probability
            if mode == 'Train':
                print(pred, target)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred_prob = torch.nn.Softmax(1)(output.data)[:, 1]
            pred_list = np.hstack((pred_list, pred_prob.cpu().view(-1).numpy()))
            target_list = np.hstack((target_list, target.cpu().view(-1).numpy()))

    roc_auc = roc_auc_score(target_list, pred_list)
    test_loss /= len(dataset_loader.dataset)

    accuracy = 100. * correct / len(dataset_loader.dataset)
    print('{:4} set: Average loss: {:.4f}, Accuracy: {:4d}/{:4d} ({:.2f}%),\t ROC-AUC: {:.2f}'.format(
        mode, test_loss, correct, len(dataset_loader.dataset),
        100. * correct / len(dataset_loader.dataset), roc_auc))

    if mode == "Val":
        epoch_val.append(epoch)
        loss_val.append(test_loss.item())
        acc_val.append(accuracy)
        roc_auc_val.append(roc_auc)
    elif mode == "Test":
        epoch_test.append(epoch)
        loss_test.append(test_loss.item())
        acc_test.append(accuracy)
        roc_auc_test.append(roc_auc)

    return accuracy


if __name__ == '__main__':
    # === ARGUMENT PARSER ===
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('--cuda', type=str, default='all', choices=['all', 'no', '0', '1'], required=True)
    args = parser.parse_args()

    if args.cuda == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    elif args.cuda == 'no':
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    
    # === KEY DEFINITIONS ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -> Model
    # model = ClassifierModel(n_classes=2)
    model = AlexNet(2)
    # model = models.DeepCoral(num_classes=2)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model = tvmodels.resnet101(pretrained=False)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = torch.nn.Linear(2048, 2, bias=True)
        def forward(self, source, target=None):
            source = self.model(source)
            return source, None
    # model = Model()
    class SuperSimple(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                # torch.nn.Linear(16384, 4096),
                torch.nn.Linear(4096, 2048),
                torch.nn.Linear(2048, 1024),
                torch.nn.Linear(1024, 2)
            )
        def forward(self, source):
            source = source.view(source.size(0), -1)
            source = self.model(source)
            return source, None
    # model = SuperSimple()

    print(utils.get_model_name(model))
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f'{torch.cuda.device_count()} GPUs used')
            model = torch.nn.DataParallel(model)
        model = model.to(device)

    # -> Optimizer
    print(f'Optimizer: {settings.opt}')
    if settings.opt == 'SGD':
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=settings.lr, momentum=settings.momentum, 
                                    weight_decay=settings.l2_decay)
    elif settings.opt == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=settings.lr, weight_decay=settings.l2_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15)
    # # lamb1 = lambda epoch: 1 / math.pow((1 + 10 * (epoch - 1) / settings.epochs), 0.75)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # -> Loss
    def nll_loss(pred, label, reduction='mean'):
        return F.nll_loss(F.log_softmax(pred, dim=1), label, reduction=reduction)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loss = torch.nn.CrossEntropyLoss()
    test_loss = torch.nn.CrossEntropyLoss(reduction='sum')

    # -> Data
    data_loader = data_loader

    epoch_val, loss_val, epoch_test, loss_test, acc_val, acc_test = [], [], [], [], [], []
    epoch_train, loss_train, roc_auc_val, roc_auc_test = [], [], [], []

    # === TRAINING ===
    for epoch in range(1, settings.epochs + 1):
        
        #train
        train(epoch, model, data_loader, optimizer, train_loss, device=device, deepcoral=settings.deepcoral)
        test(epoch, model, data_loader.train_loader, test_loss, mode="Train", device=device)
        #val
        accuracy = test(epoch, model, data_loader.val_loader, test_loss, mode="Val", device=device)
        #test
        test(epoch, model, data_loader.target_loader, test_loss, mode="Test", device=device)
        scheduler.step(accuracy.item())

    # === SAVE PLOTS AT THE END ===
    fig, ax = plt.subplots(1, 3, figsize=(20,5))
    ax[0].plot(epoch_train, loss_train, 'b', label='train')
    # ax[0].plot(epoch_val, loss_val, 'g', label='val')
    # ax[0].plot(epoch_test, loss_test, 'r', label='test')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(epoch_val, acc_val, 'g', label='val')
    ax[1].plot(epoch_test, acc_test, 'r', label='test')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    ax[2].plot(epoch_val, roc_auc_val, 'g', label='val')
    ax[2].plot(epoch_test, roc_auc_test, 'r', label='test')
    ax[2].set_title('ROC-AUC')
    ax[2].legend()

    wo = 'w/' if settings.deepcoral else 'w/o'
    fig.suptitle(f'{wo} DeepCORAL', fontsize=18)
    fig_name = f'result_plots/deepcoral{settings.deepcoral}_ep{settings.epochs}\
_opt{settings.opt}_bs{settings.batch_size}_L2{settings.l2_decay}_lr{settings.lr}.png'
    fig.savefig(fig_name, dpi=90)

    model_name = utils.get_model_name(model)
    utils.save_net(model, f'{model_name}_checkpoint.tar')
