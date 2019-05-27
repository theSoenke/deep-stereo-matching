import torch
import torch.nn as nn
from torch.autograd import Variable

class Network(nn.Module):
  def __init__(self):
    super().__init__()


    self.left = nn.Sequential( 
      nn.ReLU(),
      nn.Conv2d(1, 16*16, kernel_size=3, stride=1, padding=1),
    )
    self.right = nn.Sequential( 
      nn.ReLU(),
      nn.Conv2d(1, 16*16, kernel_size=3, stride=1, padding=1),
    )
    self.joined = nn.Sequential(
      nn.ReLU(),
      nn.Linear(65536, 256),
      nn.ReLU(),
      nn.Linear(256, 2),
      nn.Sigmoid()
    )

  def forward(self, input):      
    output_left = self.left(input[0])
    output_right = self.right(input[1])

    # Combine the two siamese networks
    x = torch.mm(output_left.view(256,256), output_right.reshape(256,256))

     # Flatten matrix into vector
    x = x.view(-1, 65536)

    return self.joined(x)