import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import cat
from torchvision import transforms as T



class blockdown(nn.Module):
    def __init__(self,inplane,outplane):
        super(blockdown,self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(inplane,outplane,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplane,outplane,3,1,1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.layer(x)
        return x


class blockup(nn.Module):
    def __init__(self,inplane,outplane):
        super(blockup, self).__init__()
        self.up = nn.ConvTranspose2d(inplane,outplane,2,2)
        self.layer = nn.Sequential(
            nn.Conv2d(inplane,outplane,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplane,outplane,3,1,1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x,cats):
        x = self.up(x)
        x = torch.cat([cats, x], dim=1)
        x = self.layer(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.encode1 = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
        )
        self.encode2 = blockdown(64,128)
        self.encode3 = blockdown(128,256)
        self.encode4 = blockdown(256,512)
        self.encode5 = blockdown(512,1024)

        self.decode1 = blockup(1024,512)
        self.decode2 = blockup(512,256)
        self.decode3 = blockup(256,128)
        self.decode4 = blockup(128,64)


    def forward(self,x):
        encode1 = self.encode1(x)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)
        encode5 = self.encode5(encode4)

        decode1 = self.decode1(encode5,encode4)
        decode2 = self.decode2(decode1,encode3)
        decode3 = self.decode3(decode2,encode2)
        decode4 = self.decode4(decode3,encode1)

        out = nn.Conv2d(64,2,1,1)(decode4)

        return out



unet = UNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(unet.parameters(),lr=0.001,momentum=0.9)
optimizer.zero_grad()
inputs = torch.randn(1,1,512,512)
# print(inputs.size())
outputs = unet(inputs)
print(outputs.size())

