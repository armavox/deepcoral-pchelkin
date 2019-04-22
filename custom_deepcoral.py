import os
import math
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader
import models_upd as models
from torch.utils import model_zoo
import settings
import utils
from classifier import ClassifierModel
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

x_train, y_train, x_test, y_test, acc_train, acc_test = [], [], [], [], [], []
    # LEARNING_RATE = settings.lr / math.pow((1 + 10 * (epoch - 1) / settings.epochs), 0.75)



def train(epoch, model, data_loader, optimizer, loss_func, device='cpu', deepcoral=False):
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print("""\n========== EPOCH {} of {} ===========
learning rate{: .6f}""".format(epoch, settings.epochs, learning_rate) )
    
    if deepcoral:
        model.train()
        iter_source = iter(data_loader.source_loader)
        iter_target = iter(data_loader.target_train_loader)
        num_iter = data_loader.len_source_loader
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, label_target = iter_target.next()
            if i % data_loader.len_target_loader == 0:
                iter_target = iter(data_loader.target_train_loader)
            data_source, label_source = data_source.to(device), label_source.to(device)
            data_target, label_target = data_target.to(device), label_target.to(device)

            optimizer.zero_grad()
            label_source_pred, loss_coral = model(data_source, data_target)
            loss_cls = loss_func(label_source_pred, label_source)
            gamma = 2  / (1 + math.exp(-10 * (epoch) / settings.epochs)) - 1
            loss_coral = torch.mean(loss_coral)
            loss = loss_cls + gamma * loss_coral
            loss.backward()
            optimizer.step()

            if i % settings.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttotal_Loss: {:.8f}\tcls_Loss: {:.8f}\tcoral_Loss: {:.8f}'.format(
                    epoch, i * len(data_source), data_loader.len_source_dataset,
                    100. * i / data_loader.len_source_loader, loss.item(), loss_cls.item(), loss_coral.item()))
    else:
        model.train()
        iter_source = iter(data_loader.source_loader)
        num_iter = data_loader.len_source_loader
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_source, label_source = Variable(data_source).to(device), Variable(label_source).to(device)

            optimizer.zero_grad()
            label_source_pred, _ = model(data_source)
            loss = loss_func(label_source_pred, label_source)
            loss.backward()
            optimizer.step()
            if i % settings.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data_source), data_loader.len_source_dataset,
                    100. * i / data_loader.len_source_loader, 
                    loss.data))

def test(model, dataset_loader, epoch, mode="training", device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = data.to(device), target.to(device)
            s_output, _ = model(data, data)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, reduction='sum').data # sumupbatchloss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataset_loader.dataset)

    accuracy = 100. * correct / len(dataset_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        mode, test_loss, correct, len(dataset_loader.dataset),
        100. * correct / len(dataset_loader.dataset)))

    if mode == "training":
        x_train.append(epoch)
        y_train.append(test_loss.item())
        acc_train.append(accuracy)
    elif mode == "testing":
        x_test.append(epoch)
        y_test.append(test_loss.item())
        acc_test.append(accuracy)

    return correct, accuracy


if __name__ == '__main__':
    # === ARGUMENT PARSER ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='SGD')
    args = parser.parse_args()

    # === CLUE DEFINITIONS ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassifierModel(n_classes=2)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f'{torch.cuda.device_count()} GPUs used')
            model = torch.nn.DataParallel(model)
        model = model.to(device)

    if settings.opt == 'SGD':
        print('Using SGD')
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=settings.lr, momentum=settings.momentum, 
                                    weight_decay=settings.l2_decay)
    elif settings.opt == 'Adam':
        print('Using Adam')
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=settings.lr, weight_decay=settings.l2_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15)
    # # lamb1 = lambda epoch: 1 / math.pow((1 + 10 * (epoch - 1) / settings.epochs), 0.75)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    def nll_loss(pred, label):
        return F.nll_loss(F.log_softmax(pred, dim=1), label)
    loss_func = nll_loss

    data_loader = data_loader

    correct = 0

    for epoch in range(1, settings.epochs + 1):
        
        train(epoch, model, data_loader, optimizer, loss_func=loss_func, device=device, deepcoral=settings.deepcoral)
        test(model, data_loader.source_loader, epoch=epoch, mode="training", device=device)
        t_correct, accuracy = test(model, data_loader.target_test_loader, epoch=epoch, mode="testing", device=device)
        scheduler.step(accuracy.item())

        # if t_correct > correct:
        #     correct = t_correct
        # print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
        #       settings.source_name, settings.target_name, correct, 100. * correct / data_loader.len_target_dataset))

    fig, ax = plt.subplots(1, 2, figsize=(16,5))
    ax[0].plot(x_train, y_train, 'g', label='train')
    ax[0].plot(x_test, y_test, 'r', label='val')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(x_train, acc_train, 'g', label='train')
    ax[1].plot(x_test, acc_test, 'r', label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    wo = 'w/' if settings.deepcoral else 'w/o'
    fig.suptitle(f'w/ DeepCORAL', fontsize=18)
    fig.savefig(f'result_plots/deepcoral{settings.deepcoral}_ep{settings.epochs}_opt{settings.opt}_bs{settings.batch_size}_L2{settings.l2_decay}_lr{settings.lr}.png', dpi=90)

    utils.save_net(model, 'checkpoint.tar')
