import torch
import torch.utils.data
from torch import optim, nn
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F
from dataset import data_set
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import data_set
from torch.autograd import Variable
from PIL import Image

class RGB2GRAY:
    def __call__(self, img):
        return img.convert("L")

batch_size = 64
learning_rate = 0.001
epochs = 20

data_transform = transforms.Compose([
    transforms.Resize([28, 28]),
    RGB2GRAY(),
    transforms.ToTensor(),
])

train_data = data_set("dataset/train", data_transform, train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_data = data_set("dataset/val", data_transform, train=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)




class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x


device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.view(-1, 28*28)
        data, target = data.to(device), target.to(device)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()
            ))

    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.to(device)
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(val_loader.dataset)
    print('\nTest set : Averge loss: {:.4f}, Accurancy: {}/{}({:.3f}%)'.format(
        test_loss, correct, len(val_loader.dataset),
        100.*correct/len(val_loader.dataset)
        ))



