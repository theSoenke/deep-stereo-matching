import network
import dataset
import utils

import torch
import torch.optim as opt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

momentum = 0.09
epochs = 14
learn_rate = 0.00003
weight_decay = 0

def main():
  kitti2015 = dataset.KITTI2015Dataset()
  cuda = torch.device('cuda')
  dataset_loader = DataLoader(dataset=kitti2015,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True)

  model = network.Network().train().cuda()
  optimizer = opt.SGD(model.parameters(), lr=learn_rate, momentum=momentum, weight_decay=weight_decay)
  #optimizer = opt.Adam(model.parameters(), lr=learn_rate, eps=1e-08, weight_decay=weight_decay)
  criterion = nn.BCELoss()

  for epoch in range(epochs):

    total_loss = 0
    for i, (images, labels) in enumerate(dataset_loader):
      # INPUT
      images[0].to(cuda)
      images[1].to(cuda)
      labels.to(cuda)

      left_images  = Variable(images[0]).cuda()
      right_images = Variable(images[1]).cuda()
      label = Variable(labels).cuda()

      #utils.show_images(images[0].data, images[1].data)

      # TRAIN
      optimizer.zero_grad()
      output = model( (left_images, right_images) )

      #print(output,label)

      loss = criterion(output, label)
      loss.backward()
      optimizer.step()

      # EVAL
      total_loss = total_loss + loss.item()

    print("["+str(i+1)+"] AvgLoss:", total_loss/i)

if __name__ == '__main__':
  main()