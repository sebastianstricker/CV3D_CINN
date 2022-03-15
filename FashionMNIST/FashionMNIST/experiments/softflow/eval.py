import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

import model
from FashionMNIST.data import unnormalize, LABEL_NAMES, NDIMS_TOTAL

from pathlib import Path


def sample_images(cinn, label, std_val, device):
    '''produces and saves to disk cINN samples for a given label (0-9)'''

    n_samples = 100

    y = torch.LongTensor(n_samples).to(device)
    y[:] = label

    z = 1.0 * torch.randn(n_samples, NDIMS_TOTAL).to(device)

    with torch.no_grad():
        samples = cinn.reverse_sample(z, y, torch.full_like(y, std_val))[0].cpu().numpy()
        samples = unnormalize(samples)
        samples = samples.mean(axis=1)

    full_image = np.zeros((28*10, 28*10))

    for k in range(n_samples):
        i, j = k // 10, k % 10
        full_image[28*i:28*(i+1), 28*j:28*(j+1)] = samples[k]

    full_image = np.clip(full_image, 0, 1)
    plt.imsave(f"eval_output/{LABEL_NAMES[label]}.png",
               full_image, vmin=0, vmax=1, cmap='gray')


def sample_images_with_temp(cinn, std_val, device):
    y = torch.arange(0, 10).repeat_interleave(7).to(device)
    temp = torch.arange(0.7, 1.4, 0.1) \
                .repeat_interleave(NDIMS_TOTAL) \
                .repeat(10) \
                .reshape((70, NDIMS_TOTAL)) \
                .to(device)
    z = temp * (torch.randn(10, NDIMS_TOTAL).repeat_interleave(7, axis=0).to(device))

    with torch.no_grad():
        samples = cinn.reverse_sample(z, y, torch.full_like(y, std_val))[0].cpu().numpy()
        samples = unnormalize(samples)
        samples = samples.mean(axis=1)

    full_image = np.zeros((28*10, 28*7))

    for k in range(70):
        i, j = k // 7, k % 7
        full_image[28*i:28*(i+1), 28*j:28*(j+1)] = samples[k]

    full_image = np.clip(full_image, 0, 1)
    plt.imsave("eval_output/temperature.png",
               full_image, vmin=0, vmax=1, cmap='gray')


def main(args):
    cinn = model.FashionMNIST_cINN(0)
    cinn.to(args.device)
    state_dict = {k: v for k, v in torch.load('output/fashionmnist_cinn.pt').items() if 'tmp_var' not in k}
    cinn.load_state_dict(state_dict)
    cinn.eval()

    Path("eval_output").mkdir(exist_ok=True)

    # Generate 100 images per class and save to disk
    for i in range(10):
        sample_images(cinn, i, args.std_val, args.device)

    # Generate 1 image per class, vary their temperature
    # and save to disk
    sample_images_with_temp(cinn, args.std_val, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        help="Device which should be used to perform evaluation",
                        choices=["cpu", "cuda"],
                        default="cuda")
    parser.add_argument("--std_val",
                        help="Conditioning standard deviation specific to softflow"
                             " used to sample images.",
                        type=float,
                        default=1e-3)
    args = parser.parse_args()
    main(args)
