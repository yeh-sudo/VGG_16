

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
])

train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        res = self.conv4(res)
        res = self.conv5(res)
        res = res.view(res.size(0), -1)
        res = self.classifier(res)
        return res


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)

summary(model, (3, 224, 224))

batch_size = 32
learning_rate = 1e-3
num_epoch = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

trainLoss = []
trainAcc = []
validAcc = []

trainSteps = len(train_loader.dataset)

model.zero_grad()
for e in range(num_epoch):
    model.train()
    train_loss = 0.0
    trainCor = 0.0
    traintotal = 0.0
    print('running epoch: %d' % (e + 1))
    print('train')
    for data, target in tqdm(train_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        trainCor += (output.argmax(1) == target).type(torch.float).sum().item()
        traintotal += target.size(0)
    trainLoss.append(train_loss / trainSteps)
    trainAcc.append(100.0 * (trainCor / traintotal))

    validCor = 0.0
    valtotal = 0.0
    with torch.no_grad():
        model.eval()
        for data, target in tqdm(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            validCor += (output.argmax(1) == target).type(torch.float).sum().item()
            valtotal += target.size(0)
        validAcc.append(100.0 * (validCor / valtotal))
    print('loss: %.3f  train acc: %.3f  valid acc: %.3f' % (train_loss / trainSteps, 100.0 * (trainCor / traintotal), 100.0 * (validCor / valtotal)))

print('Training finish')

x_axis = [*range(1, num_epoch + 1)]
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x_axis, trainAcc)
plt.plot(x_axis, validAcc)
plt.title('Accuracy', fontsize=10)
plt.ylabel('%')
plt.legend(['Training', 'Testing'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(x_axis, trainLoss)
plt.title('Model loss', fontsize=10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

torch.save(model, './model/completeModel.pth')
torch.save(model.state_dict(), './model/modelDict.pth')
