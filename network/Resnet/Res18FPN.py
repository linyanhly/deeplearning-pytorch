# construct the network for the backbone: ResNet18 and the network for the neck: FPN
# do it from scratch but refer to the pytorch of the ResNet construction
from torch import nn as nn
import torch


class bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x




class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x
