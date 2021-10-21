import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import RNN
from dataset import data_set
from torch.autograd import Variable
from PIL import Image

Epoch = 20
batch_size = 2
lr = 0.01
sequence_length=224
input_size=224
hidden_size=224
num_layers=2
num_class=2

class RGB2GRAY:
    def __call__(self, img):
        return img.convert("L")


data_transform = transforms.Compose([
    transforms.Resize([input_size, sequence_length]),
    RGB2GRAY(),
    transforms.ToTensor(),
])


model=RNN(input_size, hidden_size, num_layers, num_class, batch_size)
model.cuda()



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.5)
train_data = data_set("dataset/train", data_transform, train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_data = data_set("dataset/val", data_transform, train=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(Epoch):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        # print(images.shape)
        images=Variable(images.view(batch_size, input_size, sequence_length)).cuda()
        labels=Variable(labels).cuda()
        # print(images.shape, labels.shape)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs, labels)
        optimizer.step()

    with torch.no_grad():
        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
            %(epoch+1, Epoch, i+1, len(train_data)//batch_size, loss.data.item()))

        correct=0
        total=0
        for images, labels in train_loader: # val_loader
            # images=Variable(images.view(-1,sequence_length, input_size)).cuda()
            images=Variable(images.view(batch_size, input_size, sequence_length)).cuda()
            outputs=model(images)
            predicted= torch.max(outputs.data, 1)[1]
            total+=labels.size(0)
            correct+=(predicted.cpu()==labels).sum()

        print('test Accuracy of the model on the 2500 images: %d %%' % (100*correct/total))

torch.save(model.state_dict(), 'rnn.pkl')
