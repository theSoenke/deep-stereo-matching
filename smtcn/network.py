import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math


def cnn_out_size(out_channel, kernel_size, stride=1, padding=0, dilitation=1):
  return int(math.floor( ((out_channel + 2*padding - dilitation * (kernel_size-1)-1) / stride) + 1 ) )

class Network(nn.Module):
  def __init__(self):
    super().__init__()

    self.left = nn.Sequential( 
      nn.Conv2d(1, 32, kernel_size=5), # 32*32
      nn.ReLU(),
      nn.BatchNorm2d(32, 1e-3),

      nn.Conv2d(32, 64, kernel_size=5), # 28*28
      nn.ReLU(),
      nn.BatchNorm2d(64, 1e-3),

      nn.Conv2d(64, 64, kernel_size=5), # 24*24
      nn.ReLU(),
      nn.BatchNorm2d(64, 1e-3),

      nn.Conv2d(64, 64, kernel_size=5), # 20*20
      nn.ReLU(),
      nn.BatchNorm2d(64, 1e-3),
                  
      nn.LogSoftmax(),
    ).cuda()
    self.right = nn.Sequential( 
      nn.Conv2d(1, 32, kernel_size=5), # 32*32
      nn.ReLU(),
      nn.BatchNorm2d(32, 1e-3),

      nn.Conv2d(32, 64, kernel_size=5), # 28*28
      nn.ReLU(),
      nn.BatchNorm2d(64, 1e-3),

      nn.Conv2d(64, 64, kernel_size=5), # 24*24
      nn.ReLU(),
      nn.BatchNorm2d(64, 1e-3),

      nn.Conv2d(64, 64, kernel_size=5), # 20*20
      nn.ReLU(),
      nn.BatchNorm2d(64, 1e-3),
                  
      nn.LogSoftmax(),
    ).cuda()

    self.j_size = 128 #cnn_out_size(8, kernel_size=3)

    self.joined = nn.Sequential(
      nn.Linear( self.j_size*self.j_size, 4096),
      nn.ReLU(),
      nn.Linear(4096, 2048),
      nn.ReLU(),
      nn.Linear(2048, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Sigmoid()
    ).cuda()

  def forward(self, input):      
    output_left = self.left(input[0])
    output_right = self.right(input[1])

    # Combine the two siamese networks
    x = torch.mm( output_left.view(self.j_size,self.j_size),
                  output_right.view(self.j_size,self.j_size))

     # Flatten matrix into vector
    x = x.view(-1, self.j_size*self.j_size)

    return self.joined(x)