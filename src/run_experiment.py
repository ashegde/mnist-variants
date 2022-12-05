import torch
from torch import nn
import torch.nn.functional as F

from mnist_datamodule import MNISTDataModule
from model import ResNet
from fit import fit
from metrics import accuracy, performance

# set seed for reproducibility
torch.manual_seed(42)


def main():
    """
    Run an experiment which fits a model.
    """
    
    dir = "/content/data"
    bs = 32
    dm = MNISTDataModule(dir, batch_size = bs)
    dm.download_data()
    dm.setup()

    model = ResNet()

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    opts = {'epochs': 20, 'train_dataloader': train_dl, 'val_dataloader': val_dl, 'lr': 1e-2, 'lr_gamma': 0.7, 'wd': 1e-4}
    lossi = fit(model, opts)
    perf = performance(model, dm)


if __name__ == "__main__":
    main()
