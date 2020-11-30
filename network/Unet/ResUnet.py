import torch.nn as nn
import torchvision
import torch
from torchvision.models import resnet18

class downsample(nn.Module):
    def __init__(self,inplanes,planes):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(inplanes,planes,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes,planes,3,1,1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.down(x)
        return x

class upsample(nn.Module):
    def __init__(self,inplanes,middleplanes,planes):
        super(upsample,self).__init__()
        self.up = nn.ConvTranspose2d(inplanes,planes,2,2)
        self.conv = nn.Sequential(
            nn.Conv2d(middleplanes,planes,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes,3,1,1)
        )

    def forward(self,x,cats):
        x = self.up(x)
        x = torch.cat([cats,x],dim=1)
        out = self.conv(x)
        return out


class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet,self).__init__()
        self.resblock = list(resnet18(pretrained=True).children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,64,7,2,3),
            self.resblock[1],
            self.resblock[2]
        )
        self.layer2 = nn.Sequential(*self.resblock[3:5])
        self.layer3 = self.resblock[5]
        self.layer4 = self.resblock[6]
        self.layer5 = self.resblock[7]

        self.decode4 = upsample(512,256+256,256)
        self.decode3 = upsample(256,256+128,256)
        self.decode2 = upsample(256,128+64,128)
        self.decode1 = upsample(128,64+64,64)

        self.over = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self,x):
        down1 = self.layer1(x)
        down2 = self.layer2(down1)
        down3 = self.layer3(down2)
        down4 = self.layer4(down3)
        down5 = self.layer5(down4)

        up4 = self.decode4(down5,down4)
        up3 = self.decode3(up4,down3)
        up2 = self.decode2(up3,down2)
        up1 = self.decode1(up2,down1)

        out = self.over(up1)

        return out

resunet = ResUNet()
# print(resunet)
input = torch.randn(1,1,224,224)
optimizer = torch.optim.SGD(resunet.parameters(),0.001,0.9)
criterion = nn.CrossEntropyLoss()
optimizer.zero_grad()
outputs = resunet(input)
print(outputs.shape)


