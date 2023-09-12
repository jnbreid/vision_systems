import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


# function to count model parameters that are adjusted during training
def model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# function to make loss smooth
def smooth(f, K=5):
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]
    return smooth_f


# functions to save model/stats and load them
import os

def save_model(model, optimizer, epoch, stats, path, scheduler, iteration, best=False):
    if(not os.path.exists(path + "/models")):
        os.makedirs(path + "/models")
    if best:
        savepath = path + '/' + f"models/best_model.pth"
    else:
        savepath = path + '/' + f"models/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'stats': stats,
        'iter_': iteration
    }, savepath)
    return


def load_model(model, optimizer, scheduler, savepath):

    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    iter_ = checkpoint['iter_']

    return model, optimizer, epoch, stats, iter_

def load_model_stats(model, savepath):

    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    stats = checkpoint["stats"]

    return model, stats


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return