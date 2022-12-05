import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Any

def fit(net: nn.Module, config_opt: Dict[str, Any]) -> List:
  train_dataloader = config_opt['train_dataloader']
  val_dataloader = config_opt['val_dataloader']
  epochs = config_opt['epochs']
  lr = config_opt['lr']
  wd = config_opt['wd']
  gm = config_opt['lr_gamma']
  loss_func = F.cross_entropy

  optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gm)
  
  valiter = iter(val_dataloader)
  net.train()
  iters = 0
  outfreq = 100
  lossi=[]

  for p in net.parameters():
    p.requires_grad = True

  for epoch in range(epochs):
    for xb, yb in train_dataloader:
      # xb = (B, 28*28), yb = (B,)
      xb.unsqueeze(1) #unsqueezing in a channel dimension
      xb = xb.view(-1,1,28,28)
      yb = yb.long()
      
      logits = net(xb)
      loss = loss_func(logits, yb)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      # tracking
      lossi.append(loss.log10().item())

      # outputs
      if iters % outfreq == 0:
        with torch.no_grad():

          try:
            xv,yv = next(valiter)
          except StopIteration:
            valiter = iter(val_dataloader)
            xv,yv = next(valiter)
          
          xv.unsqueeze(1) #unsqueezing in a channel dimension
          xv = xv.view(-1,1,28,28)
          yv = yv.long()

          net.eval()
          logitsv = net(xv)
          net.train()
          lossv = loss_func(logitsv,yv)
          accv = accuracy(logitsv, yv)
          acctr = accuracy(logits, yb)
          print(f'iter {iters:7d} | epoch {epoch:2d} | loss  {loss.item():.4f} (val: {lossv.item():.4f}) | acc {acctr:.4f} (val: {accv:.4f}) | lr {scheduler.get_last_lr()[-1]:e}') 
      
      iters +=1

    scheduler.step()

  torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lossi,
            }, 'model.pt')

  return lossi