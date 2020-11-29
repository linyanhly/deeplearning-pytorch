import torch as t
import torchvision as tv
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

original_model = models.resnet18(pretrained=True)
class Resnet(nn.Module):
    def __init__(self,model):
        super(Resnet,self).__init__()
        del model.fc
        model.fc = lambda x:x
        self.features = model
        self.classifier = nn.Linear(512,10)


    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x)
        x = self.classifier(x)
        return x

model = Resnet(original_model)
print(original_model)
print(model)