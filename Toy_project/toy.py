import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
import model
import train

# hyper parameters
BATCH_SIZE = 64
TEST_BATCH_SIZE = 100
EPOCHS = 60
LEARNING_RATE = 1e-2 # 1.0
WEIGHT_DECAY=1e-5
GAMMA = 0.5
LOG_INTERVAL = 50

def test_2d(inn, test_loader):
    inn.cuda()

    #Plot initial distributions
    input_data = np.array([np.append(x.cpu().detach().numpy(), label) for x, label in test_loader.dataset])

    print("----- Initial Distributions -----")
    data.plot_distributions(input_data)

    #Plot mapping
    results = []
    with torch.no_grad():
        for x, l in test_loader:
            x = x.cuda()
            l = l.cuda()
        
            output, log_j = inn(x, l, jac=False)
            results += torch.hstack((output,torch.unsqueeze(l, dim=1)))

    results = np.array([x.cpu().detach().numpy() for x in results])

    print("----- Latent Space -----")
    data.plot_distributions(results)

    # Plot reverse samples
    results = []
    center_dist = data.GaussianDistributions(nr_samples=200, distribution_centers=[[0.0, 0.0]]*test_loader.dataset.nr_distributions)
    rev_loader  = DataLoader(center_dist, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for z, l in rev_loader:
            z = z.cuda()
            l = l.cuda()
        
            output, log_j = inn.reverse_sample(z, l)

            # Append labels to output and add to result
            results += torch.hstack((output,torch.unsqueeze(l, dim=1)))

    # From torch to numpy for plotting
    results = np.array([x.cpu().detach().numpy() for x in results])

    print("----- Sample Generation -----")
    data.plot_distributions(results)
    return

def style_transfer(inn, test_loader):
    inn.cuda()

    ###### Plot with shapes
    #Plot initial distributions
    input_data = np.array([np.append(x.cpu().detach().numpy(), label) for x, label in test_loader.dataset])

    print("----- Initial Distributions -----")
    data.plot_distributions(input_data)

    ######## style transfer
    # labels 0,..., label_count-1 will be present
    label_count = test_loader.dataset.nr_distributions
    sample_size = 5

    # as long as sample_size is low, labels will by consruction be 0
    x, l = zip(*test_loader.dataset[:sample_size])
    x = torch.stack(list(x))
    l = torch.tensor(list(l))

    # get latent vector z for samples
    with torch.no_grad():
        x = x.cuda()
        l = l.cuda()
    
        z, _ = inn(x, l, jac=False)

    # Transfer style
    results = []
    with torch.no_grad():
        for l in range(label_count):
            z = z.cuda()
            l = torch.tensor([l]*sample_size).cuda()
        
            output, _ = inn.reverse_sample(z, l)

            # Append labels to output and add to result
            results += torch.hstack((output,torch.unsqueeze(l, dim=1)))

    # From torch to numpy for plotting
    results = np.array([x.cpu().detach().numpy() for x in results])


    print("----- Style Transfer-----")
    data.plot_distributions(results)

    ####### Fix one dim to shape triangle in latent space and reverse_sample
    # dimension 1 of latent space seems to encode shape information most often.
    # Not consistently replicateable, however
    results = []
    center_dist = data.GaussianDistributions(nr_samples=50, distribution_centers=[[0.0, 0.0]]*test_loader.dataset.nr_distributions)
    rev_loader  = DataLoader(center_dist, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for z, l in rev_loader:
            
            shape_tensor = torch.full(size=(z.shape[0],1), fill_value=2.0)
            # fix dim 0
            #z = torch.hstack((shape_tensor, z))

            # fix dim 1
            d0, d2 = torch.split(z, 1, dim=1)
            z = torch.hstack((d0, shape_tensor, d2))

            # fix dim 2
            #z = torch.hstack((z, shape_tensor))

            z = z.cuda()
            l = l.cuda()
        
            output, log_j = inn.reverse_sample(z, l)

            # Append labels to output and add to result
            results += torch.hstack((output,torch.unsqueeze(l, dim=1)))

    # From torch to numpy for plotting
    results = np.array([x.cpu().detach().numpy() for x in results])

    print("----- Sample Generation - Latent Dimension 2 fixed to 2.0 -----")
    data.plot_distributions(results)
    return

def no_style(distribution_centers):
    inn = model.Toy_cINN(dims=2, nr_conditions=len(distribution_centers))

    train_data  = data.GaussianDistributions(nr_samples=800, distribution_centers=distribution_centers)
    test_data   = data.GaussianDistributions(nr_samples=400, distribution_centers=distribution_centers)

    train_loader    = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader     = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    inn = train.train(inn, train_loader, test_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, gamma=GAMMA)

    test_2d(inn, test_loader)

def with_style(distribution_centers):
    inn = model.Toy_cINN(dims=3, nr_conditions=len(distribution_centers))

    train_data  = data.GaussianDistributions(nr_samples=800, distribution_centers=distribution_centers, style=True)
    test_data   = data.GaussianDistributions(nr_samples=400, distribution_centers=distribution_centers, style=True)

    train_loader    = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader     = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    inn = train.train(inn, train_loader, test_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, gamma=GAMMA)

    style_transfer(inn, test_loader)

def main():

    distribution_centers = [[3.0,3.0], [-5.0,-5.0], [-15.0,-15.0], [12.0, 0.0]]
    
    print("######################################")
    print("Training network for Sample Generation")
    print("######################################")
    no_style(distribution_centers)

    print("#####################################")
    print("Training network for Style Transfer")
    print("#####################################")
    with_style(distribution_centers)

    return

if __name__ == "__main__":
    main()