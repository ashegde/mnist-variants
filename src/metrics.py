import torch
from torch import nn
import torch.nn.functional as F

from mnist_datamodule import MNISTDataModule

def accuracy(out: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(out, dim=1) 
    return (preds == yb).float().mean() #converts from 0-1 bool to float and then averages = the fraction of correct classifications. 

@torch.no_grad()
def performance(net: nn.Module, datamodule: MNISTDataModule):
  metrics = {'train': {}, 'val': {}, 'test': {}}

  #colab crashes due to limited RAM on the full test set, so we will reduce the test set to the training size
  # ds = {'train': (datamodule.train_ds.x, datamodule.train_ds.y),
  #       'val': (datamodule.valid_ds.x, datamodule.valid_ds.y),
  #       'test': (datamodule.test_ds.x, datamodule.test_ds.y)}
  ds = {'train': (datamodule.train_ds.x, datamodule.train_ds.y),
      'val': (datamodule.valid_ds.x, datamodule.valid_ds.y)}
  train_length = ds['train'][0].shape[0] 
  ds['test'] = (datamodule.test_ds.x[:train_length], datamodule.test_ds.y[:train_length])

  net.eval()
  for key,data in ds.items():
    x = data[0].unsqueeze(1)
    y = data[1].long()
    x = x.view(-1,1,28,28)
    logits = net(x)
    loss = F.cross_entropy(logits, y)
    acc = accuracy(logits, y) 
    metrics[key] = {'loss': loss.item(), 'acc': acc.item()}
  return metrics
