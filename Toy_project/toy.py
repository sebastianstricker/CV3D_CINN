import time

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import data

# hyper parameters
BATCH_SIZE = 64
TEST_BATCH_SIZE = 100
EPOCHS = 80
LEARNING_RATE = 1e-3 # 1.0
WEIGHT_DECAY=1e-5
GAMMA = 0.5
LOG_INTERVAL = 50

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                        nn.Linear(512,  c_out))

# def subnet_conv(c_in, c_out):
#     return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
#                         nn.Conv2d(256,  c_out, 3, padding=1))

# def subnet_conv_1x1(c_in, c_out):
#     return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(),
#                         nn.Conv2d(256,  c_out, 1))

def create_inn(dims):

    inn = Ff.SequenceINN(dims)

    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

    return inn

def plot_distributions(data):
    x1, y1 = data.T
    
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.grid(True)

    ax.scatter(x1,y1, s=plt.rcParams['lines.markersize'])

    plt.show()

def train(inn, train_loader, test_loader):
    inn.cuda()

    trainable_parameters = [p for p in inn.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=GAMMA)

    t_start = time.time()
    nll_mean = []

    dims_in = inn.dims_in[0][0]

    val_x, val_l = zip(*list(sample for sample in train_loader.dataset))
    val_x = torch.stack(val_x, 0).cuda()
    val_l = torch.LongTensor(val_l).cuda()

    print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
    for epoch in range(EPOCHS):
        for i, (x, l) in enumerate(train_loader):
            x, l = x.cuda(), l.cuda()
            z, log_j = inn(x)

            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / dims_in
            nll.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 10.)
            nll_mean.append(nll.item())
            optimizer.step()
            optimizer.zero_grad()

            if i == 0:
                with torch.no_grad():
                    z, log_j = inn(val_x)
                    nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / dims_in

                print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                                i, len(train_loader),
                                                                (time.time() - t_start)/60.,
                                                                np.mean(nll_mean),
                                                                nll_val.item(),
                                                                optimizer.param_groups[0]['lr'],
                                                                ), flush=True)
                nll_mean = []
        scheduler.step()

    return inn

def test(inn, test_loader):
    inn.cuda()
    data = np.array([x.cpu().detach().numpy() for x, label in test_loader.dataset])
    
    #plot_distributions(data)
    
    results = []
    with torch.no_grad():
        for i, (x, l) in enumerate(test_loader):
            x = x.cuda()
            l = l.cuda()
        
            output, log_j = inn(x)
            results += output

    results = np.array([x.cpu().detach().numpy() for x in results])

    plot_distributions(results)

    return

def main():

    inn = create_inn(dims=2)

    train_data  = data.GaussianDistributions(nr_samples=500)
    test_data   = data.GaussianDistributions(nr_samples=100)

    train_loader    = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader     = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

    inn = train(inn, train_loader, test_loader)

    test(inn, test_loader)

    return

if __name__ == "__main__":
    main()