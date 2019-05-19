import torch
import torch.nn as nn
import torch.optim as optim

import math

from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from dataset_loader import DatasetLoader

from itertools import cycle

import numpy as np
import settings

import matplotlib.pyplot as plt
import utils

import os
import argparse

from classifier import ClassifierModel
from original_model import AlexNet
from models_upd import DeepCoral

from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit


def train_deepcoral(epoch, model, data_loader : DatasetLoader, optimizer, loss_func, device='cpu'):
    num_iter = len(data_loader.train_data)

    global_loss = 0.0

    count = 0

    for train_batch, target_batch in zip(data_loader.train_data, cycle(data_loader.target_data)):
        data_source, label_source = train_batch
        data_target, label_target = target_batch

        i = count
        count += 1
        data_source, label_source = data_source.to(device), label_source.to(device)
        data_target, label_target = data_target.to(device), label_target.to(device)

        label_source_pred, loss_coral = model(data_source, data_target)
        loss_cls = loss_func(label_source_pred, label_source)
        gamma = np.exp(1.0 / epoch) / (1.0 + np.exp(-10.0 * (epoch) / settings.epochs))
        loss_coral = torch.mean(loss_coral)
        loss = loss_cls + gamma * loss_coral
        global_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % settings.log_interval == 0 or i == num_iter - 1:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] total_Loss: {:.8f} cls_Loss: {:.8f} coral_Loss: {:.8f} g{:.4f}'.format(
                    epoch, i, num_iter,
                           100. * i / num_iter, loss.item(), loss_cls.item(),
                           gamma * loss_coral.item(), gamma))

    return global_loss / float(num_iter)


def train_model(epoch, model, data_loader : DatasetLoader, optimizer, loss_func, device='cpu'):
    num_iter = len(data_loader.train_data)

    global_loss = 0.0
    print(len(data_loader.train_data))
#    count = 0

    for i, (data_source, label_source) in enumerate(data_loader.train_data):
        data_source, label_source = data_source.to(device), label_source.to(device)

#        i = count
#        print(i, num_iter) # count += 1

        label_source_pred, _ = model(data_source)
        loss = loss_func(label_source_pred, label_source)
        global_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % settings.log_interval == 0 or i == num_iter - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, num_iter,
                       100. * i / num_iter, loss.data))

    return global_loss / float(num_iter)


def test(model, dataset_loader, loss_func, mode="Val", device='cpu'):
    model.eval()
    test_loss = 0
    correct, total = 0, 0
    pred_list = np.array([])
    target_list = np.array([])
    with torch.no_grad():
        for i, (data, target) in enumerate(dataset_loader):
            data, target = data.to(device), target.to(device)
            if settings.deepcoral:
                output, _ = model(data)
            else:
                output, _ = model(data)
            test_loss += loss_func(output, target).item()#, reduction='sum')
            pred = output.data.max(1)[1] # get the index of the max log-probability
            # if mode == 'Val':
            #     print(pred, target)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total += target.size(0)
            pred_prob = nn.Softmax(1)(output.data)[:, 1]
            pred_list = np.hstack((pred_list, pred_prob.cpu().view(-1).numpy()))
            target_list = np.hstack((target_list, target.cpu().view(-1).numpy()))
    pred_list = np.nan_to_num(pred_list)
    roc_auc = roc_auc_score(target_list, pred_list)
    test_loss /= total
    accuracy = 100. * correct / total
    print('{:5} set: Average loss: {:.4f}, Accuracy: {:4d}/{:4d} ({:.2f}%),\t ROC-AUC: {:.2f}'.format(
        mode, test_loss, correct, total,
        100. * correct / total, roc_auc))

    return test_loss, accuracy, roc_auc


class Trainer(BaseEstimator):
    def __init__(self, num_classes, model_name, scheduler, lr_decay_epoch=10, batch_size=8, optimizer='Adam', learning_rate=0.01, l2_decay=5e-4, momentum=0.9, deepcoral=False, device='cpu'):
        print("ok")
        self.model_name = model_name
        self.model = None
        self.batch_size = batch_size
        self.l2_decay = l2_decay
        self.learning_rate = learning_rate
        self.data_loader = DatasetLoader(batch_size)
        self.momentum = momentum
        self.num_classes = num_classes

        self.deepcoral = deepcoral
        self.optimizer_name = optimizer
        self.optimizer = None
        self.device = device
        self.scheduler = scheduler
        self.lr_decay_epoch = lr_decay_epoch
        self.score_v = 0


    def __setup_model(self, num_classes):
        if self.model_name == "Alex":
            self.model = AlexNet(num_classes=num_classes)
        elif self.model_name == "Deep":
            self.model = DeepCoral(num_classes=num_classes)
        else:
            self.model = ClassifierModel(n_classes=num_classes)


        print(utils.get_model_name(self.model))
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f'{torch.cuda.device_count()} GPUs used')
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(device)

    def __create_optimizer(self, optimizer, learning_rate, l2_decay, momentum=0.9):
        self.data_loader = DatasetLoader(self.batch_size)
        opt = optim.Adam([{'params': self.model.parameters()}], lr=learning_rate, weight_decay=l2_decay)

        if optimizer == 'SGD':
            opt = optim.SGD([{'params': self.model.parameters()}],
                            lr=learning_rate, momentum=momentum,
                            weight_decay=l2_decay, nesterov=True)
        elif optimizer == 'Adagrad':
            opt = optim.Adagrad([{'params': self.model.parameters()}], lr=learning_rate, weight_decay=l2_decay)

        return opt

    def fit(self, X, y):
        self.__setup_model(self.num_classes)
        self.optimizer = self.__create_optimizer(self.optimizer_name, self.learning_rate, self.l2_decay, self.momentum)

        train_loss = nn.CrossEntropyLoss(reduction='mean')
        test_loss = nn.CrossEntropyLoss(reduction='sum')

        epoches = []
        loss_val, acc_val, roc_auc_val = [], [], []
        loss_test, acc_test, roc_auc_test = [], [], []
        loss_train  = []

        for epoch in range(1, settings.epochs + 1):
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            print("""\n========== EPOCH {} of {} ===========
            learning rate: {:.8f}""".format(epoch, settings.epochs, learning_rate))

            self.optimizer = self.scheduler(self.optimizer, epoch, init_lr=self.learning_rate, lr_decay_epoch=self.lr_decay_epoch)

            self.model.train()
            if self.deepcoral:
                loss = train_deepcoral(epoch, self.model, self.data_loader, self.optimizer, train_loss, self.device)
            else:
                loss = train_model(epoch, self.model, self.data_loader, self.optimizer, train_loss, self.device)

            loss_train.append(loss)
            test(self.model, self.data_loader.train_data, test_loss, "Train", self.device)
            l_v, a_v, r_v = test(self.model, self.data_loader.val_data, test_loss, "Val", self.device)
            l_t, a_t, r_t = test(self.model, self.data_loader.target_data, test_loss, "Test", self.device)

            loss_val.append(l_v)
            acc_val.append(a_v)
            roc_auc_val.append(r_v)

            loss_test.append(l_t)
            acc_test.append(a_t)
            roc_auc_test.append(r_t)

            epoches.append(epoch)

        fig, ax = plt.subplots(1, 3, figsize=(21, 5))
        ax[0].plot(epoches, loss_train, 'b', label='train')
        # ax[0].plot(epoch_val, loss_val, 'g', label='val')
        # ax[0].plot(epoch_test, loss_test, 'r', label='test')
        ax[0].set_title('Loss')
        ax[0].legend()

        ax[1].plot(epoches, acc_val, 'g', label='val')
        ax[1].plot(epoches, acc_test, 'r', label='test')
        ax[1].set_title('Accuracy')
        ax[1].legend()

        ax[2].plot(epoches, roc_auc_val, 'g', label='val')
        ax[2].plot(epoches, roc_auc_test, 'r', label='test')
        ax[2].set_title('ROC-AUC')
        ax[2].legend()

        wo = 'w/' if self.deepcoral else 'w/o'
        fig.suptitle(f'{wo} DeepCORAL', fontsize=17)
        model_name = utils.get_model_name(self.model)
        fig_name = f'./valid_results/{model_name}_deepcoral{self.deepcoral}_ep{settings.epochs}\
        _opt{self.optimizer_name}_bs{self.batch_size}_L2{self.l2_decay}_lr{self.learning_rate}_momentum{self.momentum}.png'
        fig.savefig(fig_name, dpi=90)
        plt.close(fig)

        self.score_v = acc_val[-1]

        return self

    def predict(self, X):
        return None

    def score(self, X, y):
        return self.score_v



if __name__ == '__main__':
    # === ARGUMENT PARSER ===
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--deepcoral', action='store_true', default=True)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-с', '--count_classes', type=int, default=2)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--cuda', type=str, default='all', choices=['all', 'no', '0', '1'], required=True)
    args = parser.parse_args()

    if args.cuda == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    elif args.cuda == 'no':
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    if args.epochs:
        settings.epochs = args.epochs

    deepcoral = args.deepcoral
    num_classes = args.count_classes

    # === KEY DEFINITIONS ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [
      "Class", # "Alex", "Deep", "Class"
    ]

    param_dist = {
        "batch_size": [4,8,16], #1, 2]
        "optimizer": ["Adam", "SGD", "Adagrad"],
        "learning_rate": np.logspace(-5,-2,10),
        "l2_decay": 5*np.logspace(-5,-1,10),
        "momentum": [0.8, 0.85, 0.9, 0.95]
    }

    n_iter_search = 35

    for model_name in models:
        X, y = [1, 2, 3], [1, 2, 3]
        clf = Trainer(num_classes, model_name, deepcoral=deepcoral, device=device, scheduler=utils.exp_lr_scheduler, lr_decay_epoch=10)
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, cv=ShuffleSplit(n_splits=1), verbose=100)
        random_search.fit(X, y)





