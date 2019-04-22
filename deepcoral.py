from __future__ import print_function
import settings
import torch
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import models_upd as models
from torch.utils import model_zoo
import matplotlib.pyplot as plt
import utils
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

training_statistic = []
testing_statistic = []
x_train, y_train, x_test, y_test, acc_train, acc_test = [], [], [], [], [], []

def load_pretrain(model):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if not "cls_fc" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    model.load_state_dict(model_dict)
    return model

def train(epoch, model, device='cpu'):
    result = []
    LEARNING_RATE = settings.lr / math.pow((1 + 10 * (epoch - 1) / settings.epochs), 0.75)
    print("""\n========== EPOCH {} of {} ===========
learning rate{: .6f}""".format(epoch, settings.epochs, LEARNING_RATE) )
    
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=LEARNING_RATE, momentum=settings.momentum, 
                                    weight_decay=settings.l2_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=LEARNING_RATE, 
                                     weight_decay=settings.l2_decay)
    model.train()
    
    iter_source = iter(data_loader.source_loader)
    iter_target = iter(data_loader.target_train_loader)
    num_iter = data_loader.len_source_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, label_target = iter_target.next()
        if i % data_loader.len_target_loader == 0:
            iter_target = iter(data_loader.target_train_loader)
        data_source, label_source = Variable(data_source).to(device), Variable(label_source).to(device)
        data_target, label_target = Variable(data_target).to(device), Variable(label_target).to(device)

        optimizer.zero_grad()
        label_source_pred, loss_coral = model(data_source, data_target)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        gamma = 2  / (1 + math.exp(-10 * (epoch) / settings.epochs)) - 1

        loss_coral = torch.mean(loss_coral)
        loss = loss_cls + gamma * loss_coral
        loss.backward()
        optimizer.step()
        if i % settings.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttotal_Loss: {:.8f}\tcls_Loss: {:.8f}\tcoral_Loss: {:.8f}'.format(
                epoch, i * len(data_source), data_loader.len_source_dataset,
                100. * i / data_loader.len_source_loader, loss.item(), loss_cls.item(), loss_coral.item()))

        result.append({
            'epoch': epoch,
            'step': i + 1,
            'total_steps': num_iter,
            'loss': loss.item(),
            'cls loss': loss_cls.item(),
            'coral loss': loss_coral.item()
        })

    return result

def test(model, dataset_loader, epoch, mode="training", device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            s_output, _ = model(data, data)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, reduction='sum').data # sumupbatchloss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataset_loader.dataset)

    accuracy = 100. * correct / len(dataset_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        mode, test_loss, correct, len(dataset_loader.dataset),
        100. * correct / len(dataset_loader.dataset)))

    testing_statistic.append({
        'data': mode,
        'epoch': epoch,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(dataset_loader.dataset),
        'accuracy': accuracy
    })

    if mode == "training":
        x_train.append(epoch)
        y_train.append(test_loss.item())
        acc_train.append(accuracy)
    elif mode == "testing":
        x_test.append(epoch)
        y_test.append(test_loss.item())
        acc_test.append(accuracy)

    return correct


if __name__ == '__main__':
    model = models.DeepCoral(num_classes=2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='SGD', 
                    choices=['SGD', 'Adam'])
    global args
    args = parser.parse_args()

    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f'{torch.cuda.device_count()} GPUs used')
            model = torch.nn.DataParallel(model)
        model = model.to(device)
    # print(model)

    for epoch in range(1, settings.epochs + 1):
        res = train(epoch, model, device=device)
        training_statistic.append(res)

        test(model, data_loader.source_loader, epoch=epoch, mode="training", device=device)
        t_correct = test(model, data_loader.target_test_loader, epoch=epoch, mode="testing", device=device)

        if t_correct > correct:
            correct = t_correct
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              settings.source_name, settings.target_name, correct, 100. * correct / data_loader.len_target_dataset))

    fig, ax = plt.subplots(1, 2, figsize=(16,5))
    ax[0].plot(x_train, y_train, 'g', label='train')
    ax[0].plot(x_test, y_test, 'r', label='val')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(x_train, acc_train, 'g', label='train')
    ax[1].plot(x_test, acc_test, 'r', label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    fig.suptitle('ResNet50 w/ DeepCORAL', fontsize=18)
    fig.savefig(f'result_plots/coral_loss_ep{settings.epochs}_opt{args.opt}_bs{settings.batch_size}_L2{settings.l2_decay}_lr{settings.lr}.png', dpi=90)

    utils.save_net(model, 'checkpoint.tar')
