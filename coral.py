import torch
import numpy as np

def CORAL(source, target):
    d = source.data.shape[0]  # Batch size
    assert source[0].shape == target[0].shape, f'src.sh is {source.shape}; tg.sh is {target.shape}!'

    # Source covariance
    source -= torch.mean(source, 1, keepdim=True)
    source_cov = torch.matmul(torch.transpose(source, -2,-1), source)
    # source_cov = torch.mean(source_cov, dim=(-2,-1))
    # Target covariance
    target -= torch.mean(target, 1, keepdim=True)
    target_cov = torch.matmul(torch.transpose(target, -2, -1), target)

    loss = torch.norm((source_cov - target_cov), dim=(-2,-1))
    loss = loss/(4*d**2)
    loss = torch.mean(loss)
    return loss

    # xm = torch.mean(source, 1, keepdim=True) - source
    # xc = torch.matmul(torch.transpose(xm, 0, 1), xm)
    # # target covariance
    # xmt = torch.mean(target, 1, keepdim=True) - target
    # xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # # frobenius norm between source and target
    # loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
