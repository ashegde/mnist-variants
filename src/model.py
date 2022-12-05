"""Basic convolutional model building blocks."""
import torch
from torch import nn
import torch.nn.functional as F

CONV_DIM = 64
FC_DIM = 256


class ResBlock(nn.Module):
    """
    Residual block with two 3x3 convolutional layers
    Padding size 1 to preserve input dimensionality
    ReLU nonlinearity
    """

    def __init__(self, indim: int, outdim: int, ksize=3, s=1) -> None:
        super().__init__()
        #{in,out}dim = number of filters (each of size ksize_in) in the layer


        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=ksize, stride=s, padding=1)
        self.bn = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=ksize, stride=s, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """.

        Inputs
        ----------
        x
            (B, C_in, H_in, W_in) tensor

        Outputs
        -------
        torch.Tensor
            (B, C_out, H_out, W_out) tensor
        """
        z = self.conv1(x)  # (B, C_out, H_in, W_in)
        z = self.bn(z)     # (B, C_out, H_in, W_in)  
        z = self.relu(z)   # (B, C_out, H_in, W_in)
        z = self.conv2(z)  # (B, C_out , H_in, W_in) 
        outs = x+z
        return outs

class ResNet(nn.Module):
    """
    Simple ResNet for the Rotated MNIST dataset
    Recall, the MNIST dataset takes as input (channels, width, height) = (1,28,28) images, i.e., 784 dimensional feature vectors
    and has 10 classes (for the digits 0,1,...,9)
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        #hardwired for MNIST
        self.input_height, self.input_width = 28,28 
        num_classes = 10

        conv_dim = self.args.get("conv_dim", CONV_DIM) #number of filters / channels
        fc_dim = self.args.get("fc_dim", FC_DIM)


        ## input = (B,C=1,H,W), C=1 as MNIST has just a single channel
        self.res1 = ResBlock(1, conv_dim)   
        self.res2 = ResBlock(conv_dim, conv_dim)
        self.max_pool = nn.MaxPool2d(2)
        self.res3 = ResBlock(conv_dim, conv_dim)
        self.res4 = ResBlock(conv_dim, conv_dim)
        self.res5 = ResBlock(conv_dim, conv_dim)
        self.res6 = ResBlock(conv_dim, conv_dim)
        self.bn2d = nn.BatchNorm2d(conv_dim)
        self.drop = nn.Dropout(0.5)
        conv_output_height, conv_output_width = self.input_height // 4, self.input_width // 4
        self.flatten = nn.Flatten()
        fc_input_dim = int(conv_output_height * conv_output_width * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.bn1d = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        self.relu = nn.ReLU()

        #adjust inits
        with torch.no_grad():
          self.fc2.weight *= 0.001
          self.fc2.bias *= 0.001
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, Ch, H, W) tensor, where H and W must equal input height and width from data_config.

        Returns
        -------
        torch.Tensor
            (B, Classes) tensor
        """
        B, C, H, W = x.shape
        x = self.res1(x)      # (B, CONV_DIM, H, W)
        x = self.res2(x)      # (B, CONV_DIM, H, W)
        x = self.max_pool(x)  # (B, CONV_DIM, H // 2, W // 2)
        x = self.drop(x)
        x = self.res3(x)      # (B, CONV_DIM, H // 2, W // 2)
        x = self.res4(x)      # (B, CONV_DIM, H // 2, W // 2)
        x = self.max_pool(x)  # (B, CONV_DIM, H // 4, W // 4)
        x = self.drop(x)
        x = self.res5(x)      # (B, CONV_DIM, H // 4, W // 4)
        x = self.res6(x)      # (B, CONV_DIM, H // 4, W // 4)
        x = self.bn2d(x)
        x = self.relu(x)
        x = self.flatten(x)   # (B, CONV_DIM * H // 4 * W // 4)
        x = self.fc1(x)       # (B, FC_DIM)
        x = self.bn1d(x)      # (B, FC_DIM)
        x = self.relu(x)      # (B, FC_DIM)
        x = self.fc2(x)       # (B, Classes)
        return x