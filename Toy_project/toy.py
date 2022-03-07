from random import gauss
import torch
import torch.nn as nn

import numpy as np

from matplotlib import pyplot as plt

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import time

# hyper parameters
BATCH_SIZE = 64
TEST_BATCH_SIZE = 100
EPOCHS = 10
LEARNING_RATE = 1e-4 # 1.0
WEIGHT_DECAY=1e-5
GAMMA = 0.1
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

def plot_distributions(gauss1, gauss2):
    x1, y1 = gauss1.T
    x2, y2 = gauss2.T
    
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.grid(True)

    ax.scatter(x1,y1, s=plt.rcParams['lines.markersize'])
    ax.scatter(x2,y2, s=plt.rcParams['lines.markersize'])

    plt.show()

def train(inn, train_data, test_data):
    inn.cuda()

    trainable_parameters = [p for p in inn.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40])
    
    N_epochs = 60
    t_start = time.time()
    nll_mean = []

    dims_in = inn.dims_in[0][0]

    for epoch in range(N_epochs):
        for i, (x, l) in enumerate(train_data):
            x, l = x.cuda(), l.cuda()
            z, log_j = inn(x, l)

            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / dims_in
            nll.backward()
            torch.nn.utils.clip_grad_norm_(inn.trainable_parameters, 10.)
            nll_mean.append(nll.item())
            inn.optimizer.step()
            inn.optimizer.zero_grad()

            if not i % LOG_INTERVAL:
                with torch.no_grad():
                    z, log_j = inn(list(zip(*train_data)))
                    nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / dims_in

                print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                                i, 100,
                                                                (time() - t_start)/60.,
                                                                np.mean(nll_mean),
                                                                nll_val.item(),
                                                                inn.optimizer.param_groups[0]['lr'],
                                                                ), flush=True)
                nll_mean = []
        scheduler.step()
    return

def test(inn, gauss1, gauss2):
    return

def generate_data():

    gauss_1 = torch.tensor(np.random.multivariate_normal(mean=[-2, -5], cov=np.eye(2), size=500))
    gauss_2 = torch.tensor(np.random.multivariate_normal(mean=[5, 3], cov=np.eye(2), size=500))
    
    #plot_distributions(gauss_1, gauss_2)

    l1 = torch.zeros(len(gauss_1))
    l2 = torch.ones(len(gauss_2))

    train_data   = zip(torch.vstack((gauss_1[400:], gauss_2[400:])), torch.hstack((l1[400:], l2[400:])))
    test_data    = zip(torch.vstack((gauss_1[400:], gauss_2[400:])), torch.hstack((l1[400:], l2[400:])))

    return train_data, test_data

def main():

    inn = create_inn(dims=2)

    train_data, test_data = generate_data()

    train(inn, train_data, test_data)

    test(inn, test_data)

    return

if __name__ == "__main__":
    main()