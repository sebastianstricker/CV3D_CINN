from collections import defaultdict
import numpy as np

import torch
from torch.utils.data       import Dataset
from torch.distributions    import MultivariateNormal

from matplotlib import pyplot as plt
from matplotlib import patches as plot_patches

markers = ['o', 's', '^'] # circle, square, triangle
colors = ['red', 'green', 'blue', 'pink', 'black', 'purple']

class GaussianDistributions(Dataset):

    def __init__(self, nr_samples, distribution_centers, style=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.nr_samples = nr_samples
        self.nr_distributions = len(distribution_centers)

        # Create a distribution for each given center point
        distributions = [MultivariateNormal(torch.tensor(center), torch.eye(len(center))) for center in distribution_centers]
        
        # Create samples and labels for each distribution
        self.data = []
        shapes = torch.tensor([0,1,2])

        for label, distribution in enumerate(distributions):
            for i in range(nr_samples):
                sample = distribution.sample()

                if style:
                    shape = torch.randint(low=0, high=len(shapes), size=(1,))
                    sample = torch.cat((sample, shape))
                self.data.append((sample, label))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]

        return sample

def plot_distributions(data):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    # Only plot datapoints
    if len(data[0]) == 2:
        x1,y1 = data.T
        ax.scatter(x1,y1, s=plt.rcParams['lines.markersize'])

    # Plot datapoints, color by labelling
    elif len(data[0]) == 3:
        legend_handles = []

        # group data by label
        label_dict = defaultdict(list)
        for x, y, label in data:
            label_dict[label].append([x,y])
        
        # plot with color by label
        for label, datapoints in label_dict.items():
            legend_handles.append(plot_patches.Patch(color=colors[int(label)], label=f'label {int(label)}'))

            x1,y1 = np.array(datapoints).T
            ax.scatter(x1,y1, s=plt.rcParams['lines.markersize'], label=f'label {int(label)}', color=colors[int(label)]) 
       
        ax.legend(handles=legend_handles)

    # Plot datapoints, color & shape
    elif len(data[0]) == 4:
        invalid_shape_count = 0
        legend_handles = []

        # group data by label
        label_dict = defaultdict(list)
        for x, y, shape, label in data:
            label_dict[label].append([x,y, shape])
        
        for label, group in label_dict.items():
            legend_handles.append(plot_patches.Patch(color=colors[int(label)], label=f'label {int(label)}'))

            # group data by shape
            shape_dict = defaultdict(list)
            for x, y, shape in group:
                s = round(shape)
                if s < 0 or s > 2:
                    invalid_shape_count += 1
                    continue
                shape_dict[s].append([x,y])

            for shape, datapoints in shape_dict.items():
                # plot with color by label and shape
                x1,y1 = np.array(datapoints).T
                ax.scatter(x1,y1, s=plt.rcParams['lines.markersize'], label=f'label {int(label)}', marker=markers[shape], color=colors[int(label)])
       
        ax.legend(handles=legend_handles)
        print(f"Shape class invalid: {str(invalid_shape_count)} of {len(data)}")

    plt.show()