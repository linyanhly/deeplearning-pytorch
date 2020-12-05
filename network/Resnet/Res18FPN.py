# construct the network for the backbone: ResNet18 and the network for the neck: FPN
# do it from scratch but refer to the pytorch of the ResNet construction

# 2020/12/2 ,completing the network combining the torchvision models,
# I think Resnet model with fpn can be completed after today

# 2020/12/5,completed it with resnet18 from torchvision

from torch import nn as nn
from torchvision.models import resnet18
import torch
import torch.nn.functional as F


class Resfpn(nn.Module):
    def __init__(self):
        super(Resfpn, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resblock = list(self.resnet.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            self.resblock[1],
            self.resblock[2],
        )
        self.layer2 = nn.Sequential(*self.resblock[3:5])
        self.layer3 = self.resblock[5]
        self.layer4 = self.resblock[6]
        self.layer5 = self.resblock[7]

        self.top_layer = nn.Conv2d(512, 64, 1, 1, 0)

        self.decode4 = nn.Conv2d(256, 64, 1, 1, 0)
        self.decode3 = nn.Conv2d(128, 64, 1, 1, 0)
        self.decode2 = nn.Conv2d(64, 64, 1, 1, 0)

        self.smooth = nn.Conv2d(64, 64, 3, 1, 1)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W)) + y

    def forward(self, x):
        C1 = self.layer1(x)
        C2 = self.layer2(C1)
        C3 = self.layer3(C2)
        C4 = self.layer4(C3)
        C5 = self.layer5(C4)

        M5 = self.top_layer(C5)
        M4 = self.upsample_add(M5, self.decode4(C4))
        M3 = self.upsample_add(M4, self.decode3(C3))
        M2 = self.upsample_add(M3, self.decode2(C2))

        P5 = self.smooth(M5)
        P4 = self.smooth(M4)
        P3 = self.smooth(M3)
        P2 = self.smooth(M2)

        return P2, P3, P4, P5


res18fpn = Resfpn()
# print(res18fpn)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(res18fpn.parameters(), 0.001, 0.9)
optimizer.zero_grad()
inputs = torch.randn(1, 1, 224, 224)
P2, P3, P4, P5 = res18fpn(inputs)

print(P2.size(), P3.size(), P4.size(), P5.size())

