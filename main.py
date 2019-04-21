import models
import settings
import data_loader
import utils
import argparse
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

testing_statistic = []
plot_epochs_train, y_train, plot_epochs_test, y_test, acc_train, acc_test = [], [], [], [], [], []

def train(epoch, model, device='cpu'):
    result = []
    LEARNING_RATE = settings.lr / math.pow((1 + 10 * (epoch - 1) / settings.epochs), 0.75)
    print("""\n\n========== EPOCH {} of {} ===========
learning rate{: .6f}""".format(epoch, settings.epochs, LEARNING_RATE) )

    if args.opt == 'SGD':
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=LEARNING_RATE, momentum=settings.momentum, 
                                    weight_decay=settings.l2_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=LEARNING_RATE, 
                                     weight_decay=settings.l2_decay)

    model.train()
    iter_source = iter(data_loader.source_loader)
    num_iter = data_loader.len_source_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_source, label_source = Variable(data_source).to(device), Variable(label_source).to(device)

        optimizer.zero_grad()
        label_source_pred = model(data_source)
        loss = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        loss.backward()
        optimizer.step()
        if i % settings.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data_source), data_loader.len_source_dataset,
                100. * i / data_loader.len_source_loader, 
                loss.data))

        result.append({
            'epoch': epoch,
            'step': i + 1,
            'total_steps': num_iter,
            'loss': loss.data,  # classification_loss.data[0]
        })

    return result


def test(model, dataset_loader, epoch, mode="training", device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            s_output = model(data)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, reduction='sum').data # sumup batchloss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataset_loader.dataset)
    accuracy = 100. * correct / len(dataset_loader.dataset)
    print("""Mode: {}, Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)"""
        .format(
        mode, epoch, test_loss, correct, len(dataset_loader.dataset), accuracy))

    testing_statistic.append({
        'data': mode,
        'epoch': epoch,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(dataset_loader.dataset),
        'accuracy':  accuracy
    })

    if mode == "training":
        plot_epochs_train.append(epoch)
        y_train.append(test_loss.item())
        acc_train.append(accuracy)
    elif mode == "testing":
        plot_epochs_test.append(epoch)
        y_test.append(test_loss.item())
        acc_test.append(accuracy)

    return correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='SGD', 
                    choices=['SGD', 'Adam'])
    global args
    args = parser.parse_args()

    models.resNet_main = True
    model = models.resnet50(settings.use_checkpoint)
    # if torch.cuda.is_available():
    #     model.cuda()
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)
   #  print(model)

    training_statistic = []

    for epoch in range(1, settings.epochs + 1):
        res = train(epoch, model, device)
        training_statistic.append(res)

        test(model, data_loader.source_loader, epoch=epoch, mode="training", device=device)
        t_correct = test(model, data_loader.target_test_loader, epoch=epoch, mode="testing", device=device)

        if t_correct > correct:
            correct = t_correct
        print('source: {} max correct: {} max target accuracy{: .2f}%\n'.format(
              settings.source_name, correct, 100. * correct / data_loader.len_target_dataset))

    fig, ax = plt.subplots(1, 2, figsize=(16,5))
    ax[0].plot(plot_epochs_train, y_train, 'g', label='train')
    ax[0].plot(plot_epochs_test, y_test, 'r', label='val')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(plot_epochs_train, acc_train, 'g', label='train')
    ax[1].plot(plot_epochs_test, acc_test, 'r', label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    fig.suptitle('ResNet50 w/o DeepCORAL', fontsize=18)
    fig.savefig(f'result_plots/resnet50_ep{settings.epochs}_opt{args.opt}_bs{settings.batch_size}_L2{settings.l2_decay}_lr{settings.lr}.png', dpi=90)

    # utils.save(training_statistic, 'training_statistic.pkl')
    # utils.save(testing_statistic, 'testing_statistic.pkl')
    utils.save_net(model, 'checkpoint.tar')
