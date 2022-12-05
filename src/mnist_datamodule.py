import requests
import zipfile
import os
import torch


class BaseDataset(torch.utils.data.Dataset):

  def __init__(self, x,y):
    super().__init__()
    self.x = x
    self.y = y

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, index):
    'produces a single data point'
    xs = self.x[index]
    ys = self.y[index]

    return xs, ys

class MNISTDataModule:
  '''
  Helper module that downloads, loads, and partitions the MNIST dataset.
  '''
  url = "http://www.iro.umontreal.ca/~lisa/icml2007data/"
  filename = "mnist_rotation_back_image_new.zip"

  def __init__(self, dir, batch_size=32):
      self.dir = dir 
      self.batch_size = batch_size
      self.path = self.dir + '/' +self.filename

  def download_data(self):
    # create directories and download dataset
    if not os.path.exists(self.dir):
      os.mkdir(self.dir)
    if not os.path.exists(self.path):
      content = requests.get(self.url + self.filename).content
      with open(self.path, "wb") as f:
        f.write(content)
    with zipfile.ZipFile(self.path) as f:
      f.extractall(path=self.dir)
      
  def setup(self):
    # load data
    with open(self.dir+'/mnist_all_background_images_rotation_normalized_test.amat', 'r') as f1:
      ds_te = [[float(a) for a in line.split()] for line in f1]
    with open(self.dir+'/mnist_all_background_images_rotation_normalized_train_valid.amat', 'r') as f2:
      ds_tr_val = [[float(a) for a in line.split()]  for line in f2]
      
    ds_te, ds_tr_val = map(torch.tensor, (ds_te, ds_tr_val))

    # hardwired 80%-20% split into training and validation
    n1 = int(0.8*ds_tr_val.shape[0])

    Xtr, Ytr = ds_tr_val[:n1,:-1], ds_tr_val[:n1,-1]
    Xval, Yval = ds_tr_val[n1:,:-1], ds_tr_val[n1:,-1]
    Xte, Yte = ds_te[:,:-1], ds_te[:,-1]
    
    self.train_ds = BaseDataset(Xtr, Ytr)
    self.valid_ds = BaseDataset(Xval, Yval)
    self.test_ds = BaseDataset(Xte, Yte)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.valid_ds, batch_size=3*self.batch_size, shuffle=False)