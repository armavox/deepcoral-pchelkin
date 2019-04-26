import pickle
import torch
import settings as s

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))


def save_net(model, path):
    torch.save(model.state_dict(), path)
    print('[INFO] Checkpoint saved to {}'.format(path))


def load_net(model, path):
    model.load_state_dict(torch.load(path))
    print('[INFO] Checkpoint {} loaded'.format(path))

def get_model_name(model):
    r = model.__repr__()
    return r[:r.find('(')]


def exp_lr_scheduler(optimizer, epoch, init_lr=s.lr, lr_decay_epoch=s.lr_decay_epoch):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (s.lr_decay_weight**(epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer