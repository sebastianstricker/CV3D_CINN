import json
import torch

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from time import time

from FashionMNIST.utils import parse_config

from model import FashionMNIST_cINN
from data import get_trainval_datasets, get_dataloader, unnormalize, \
                              NDIMS_TOTAL, IM_DIMS


def main(config):

    """
    Prepare data
    """

    data_config = parse_config(config, "data", ["seed", "batch_size", "device"])
    train_ds, val_ds = get_trainval_datasets(data_config)
    train_dl = get_dataloader(train_ds, data_config)
    val_dl = get_dataloader(val_ds, data_config)

    """
    Prepare model
    """

    train_config = parse_config(config, "train", ["seed", "batch_size", "device"])
    cinn = FashionMNIST_cINN(train_config["lr"])
    cinn.to(train_config["device"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, **train_config["scheduler"])

    """
    Prepare output directories
    """

    (Path(__file__).parent / "train_output").mkdir(exist_ok=True)

    """
    Perform training

    Code adapted from: https://github.com/VLL-HD/conditional_INNs/blob/master/mnist_minimal_example/train.py
    """

    t_start = time()
    nll_mean = []

    print("Epoch\tBatch/Total\tTime\tNLL train\tNLL val\tLR")
    for epoch in range(train_config["n_epochs"]):
        for i, (x, y) in enumerate(train_dl):

            x, y = x.to(train_config["device"]), y.to(train_config["device"])
            z, log_j = cinn(x, y)

            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / NDIMS_TOTAL
            nll.backward()
            torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
            nll_mean.append(nll.item())
            cinn.optimizer.step()
            cinn.optimizer.zero_grad()

            if not i % 50:
                with torch.no_grad():

                    nll_val_mean = []
                    for (x_val, y_val) in val_dl:
                        x_val, y_val = x_val.to(train_config["device"]), y_val.to(train_config["device"])
                        z, log_j = cinn(x_val, y_val)
                        nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / NDIMS_TOTAL
                        nll_val_mean.append(nll_val.item())

                print("%.3i\t%.5i/%.5i\t%.2f\t%.6f\t%.6f\t%.2e" % (epoch,
                                                                   i, len(train_dl),
                                                                   (time() - t_start)/60.,
                                                                   np.mean(nll_mean),
                                                                   np.mean(nll_val_mean),
                                                                   cinn.optimizer.param_groups[0]["lr"],
                                                                   ), flush=True)
                nll_mean = []

        # Store 10 generated images per class every 10th epoch
        if not epoch % 10:
            y = torch.arange(0, 10).repeat_interleave(10).to(train_config["device"])
            z = z = 1.0 * torch.randn(100, NDIMS_TOTAL).to(train_config["device"])

            with torch.no_grad():
                samples = cinn.reverse_sample(z, y)[0].cpu().numpy()
                samples = unnormalize(samples)
                if IM_DIMS[0] > 1:
                    samples = samples.mean(axis=1)

            full_image = np.zeros((28*10, 28*10))

            for k in range(100):
                i, j = k // 10, k % 10
                full_image[28*i:28*(i+1), 28*j:28*(j+1)] = samples[k]

            full_image = np.clip(full_image, 0, 1)
            plt.imsave(Path(__file__).parent / "train_output" / f"epoch_{epoch}.png",
                       full_image, vmin=0, vmax=1, cmap="gray")

        scheduler.step()

    Path("output").mkdir(exist_ok=True)
    torch.save(cinn.state_dict(), 'output/fashionmnist_cinn.pt')


if __name__ == "__main__":
    with open(Path(__file__).parent / "config.json", "r", encoding="utf8") as f:
        config = json.load(f)
    main(config)
