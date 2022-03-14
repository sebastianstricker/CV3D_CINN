import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import time
"""
Adapted from: https://github.com/VLL-HD/conditional_INNs/blob/master/mnist_minimal_example
"""
def train(inn, train_loader, test_loader, learning_rate, weight_decay, gamma, epochs):
    inn.cuda()

    trainable_parameters = [p for p in inn.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(0,epochs,10)], gamma=gamma)

    val_x, val_l = zip(*list(sample for sample in test_loader.dataset))
    val_x = torch.stack(val_x, 0).cuda()
    val_l = torch.LongTensor(val_l).cuda()

    t_start = time.time()
    nll_list = []
    print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\t\tLR')
    for epoch in range(epochs):
        for batch, (x, l) in enumerate(train_loader):
            x, l = x.cuda(), l.cuda()

            z, log_j = inn(x, l)

            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / inn.dims
            nll.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 10.)
            nll_list.append(nll.item())
            optimizer.step()
            optimizer.zero_grad()

            if not batch % 50:
                curr_lr = optimizer.param_groups[0]['lr']
                print_training_progress(inn, val_x, val_l, t_start, epoch, batch, len(train_loader), np.mean(nll_list), curr_lr)
                nll_list = []
        scheduler.step()

    return inn

def print_training_progress(inn, testdata_x, testdata_l, t_start, epoch, batch, nr_batches, nll_mean, curr_lr):
    with torch.no_grad():
        z, log_j = inn(testdata_x, testdata_l)
        nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / inn.dims

    print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                    batch, nr_batches,
                                                    (time.time() - t_start)/60.,
                                                    nll_mean,
                                                    nll_val.item(),
                                                    curr_lr,
                                                    ), flush=True)