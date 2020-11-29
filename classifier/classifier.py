import torch as t
import torchvision as tv

class Net(t.nn.Module):
    '''
    定义一个network，从而在后面调用该network
    '''
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = t.nn.Conv2d(3,6,(5,5))
        self.pool = t.nn.MaxPool2d(2,2)
        self.conv2 = t.nn.Conv2d(6,16,3)
        self.fc1 = t.nn.Linear(16*6*6,10)

    def forward(self,x):
        x = self.pool(t.nn.functional.relu(self.conv1(x)))
        x = self.pool(t.nn.functional.relu(self.conv2(x)))
        x = x.view(-1,16*6*6)
        x = t.sigmoid(self.fc1(x))
        return x

def dataloader():

    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(),
         tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

    trainset = tv.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
    testset = tv.datasets.CIFAR10(root='./data',train=False,download=True)

    trainloader = t.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)
    testloader = t.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

    return trainloader,testloader



class Config():

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')



import matplotlib.pyplot as plt
import numpy as np
class EDA():
    '''
    展示CIFAR10的图片，通过该图片观察CIFAR10的具体
    '''
    def imshow(self,img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        npimg = np.transpose(npimg,(1,2,0))
        plt.imshow(npimg)
        plt.show()

trainloader,testloader = dataloader()
dataiter = iter(trainloader)
images,labels = dataiter.next()
eda = EDA()
eda.imshow(tv.utils.make_grid(images))
print(' '.join('%5s' % Config().classes[labels[j]] for j in range(4)))

net = Net()

def metric():
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    return criterion,optimizer

criterion, optimizer = metric()

for epoch in range(2):

    running_loss = 0.0
    for i,data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch+1,i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished training")

