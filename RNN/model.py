import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable




class RNN(nn.Module):
     def __init__(self, input_size, hidden_size, num_layers, num_classes, BS):
         super(RNN, self).__init__()
         self.hidden_size = hidden_size
         self.num_layers = num_layers
         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
         self.fc = nn.Linear(hidden_size, num_classes)


     def forward(self, x):
         # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
         
         # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
         h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
         c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

         # Forward propagate RNN
         out, (_, _) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)

         # Decode hidden state of last time step
         out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
         return out
