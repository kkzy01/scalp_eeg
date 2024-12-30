import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.net = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=(1,9), stride=(1,stride), padding=(0,4), bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU(),
        nn.Conv2d(planes, planes, kernel_size=(1,9), stride=1, padding=(0,4), bias=False),
        nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                        kernel_size=1, stride=(1,stride), bias=False),
                nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = self.net(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNN2D_LSTM_V8_4(nn.Module):
        def __init__(self,device,num_layers=2,dropout=0.1,batch_size=32,):
                super(CNN2D_LSTM_V8_4, self).__init__()      
                self.device = device
                self.num_layers = num_layers
                self.hidden_dim = 256
                self.dropout = dropout
                self.batch_size = batch_size
                self.num_data_channel = 1
                self.in_planes = 64
                
                activation = 'relu'
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()],
                        ['relu', nn.ReLU(inplace=True)],
                        ['tanh', nn.Tanh()],
                        ['sigmoid', nn.Sigmoid()],
                        ['leaky_relu', nn.LeakyReLU(0.2)],
                        ['elu', nn.ELU()]
                ])

                # Create a new variable for the hidden state, necessary to calculate the gradients
                self.hidden = ((torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)))
                
                block = BasicBlock

                def conv2d_bn(inp, oup, kernel_size, stride, padding, dilation=1):
                        return nn.Sequential(
                                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                                nn.BatchNorm2d(oup),
                                self.activations[activation],
                )
                def conv2d_bn_nodr(inp, oup, kernel_size, stride, padding):
                        return nn.Sequential(
                                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.BatchNorm2d(oup),
                                self.activations[activation],
                )  
                self.conv1 = conv2d_bn(self.num_data_channel,  64, (1,51), (1,4), (0,25))
                self.maxpool1 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))

                self.layer1 = self._make_layer(block, 64, 2, stride=1)
                self.layer2 = self._make_layer(block, 128, 2, stride=2)
                self.layer3 = self._make_layer(block, 256, 2, stride=2)

                self.agvpool = nn.AdaptiveAvgPool2d((1,1))

                self.lstm = nn.LSTM(
                        input_size=256,
                        hidden_size=self.hidden_dim,
                        num_layers=self.num_layers,
                        batch_first=True,
                        dropout=self.dropout) 

                self.classifier = nn.Sequential(
                        nn.Linear(in_features=self.hidden_dim, out_features= 64, bias=True),
                        nn.BatchNorm1d(64),
                        self.activations[activation],
                        nn.Linear(in_features=64, out_features= 2, bias=True),
                )

        def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for stride1 in strides:
                        layers.append(block(self.in_planes, planes, stride1))
                        self.in_planes = planes
                return nn.Sequential(*layers)
        
        def forward(self, x):
                x.permute(0,2,1)
                x = x.unsqueeze(1)
                x = self.conv1(x)
                x = self.maxpool1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.agvpool(x)
                x = torch.squeeze(x, 2)
                x = x.permute(0, 2, 1)
                self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
                lstm_out, self.hidden = self.lstm(x, self.hidden)

                # Take the last output for classification
                lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
                out = self.classifier(lstm_out)
                return out
                
        def init_state(self, device, batch_size):
                self.hidden = ((torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)))

