import torch
from torch.utils.data       import Dataset
from torch.distributions    import MultivariateNormal

distribution_centers = [[3.0,3.0], [-5.0,-5.0]]

class GaussianDistributions(Dataset):

    def __init__(self, nr_samples, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.nr_samples = nr_samples
        self.transform = transform

        # Create a distribution for each given center point
        distributions = [MultivariateNormal(torch.tensor(center), torch.eye(2)) for center in distribution_centers]
        
        # Create samples and labels for each distribution
        self.data = []
        for label, distribution in enumerate(distributions):
            for i in range(nr_samples):
                self.data.append((distribution.sample(), label))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample