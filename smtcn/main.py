import network
import dataset

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


def main():
  kitti2015 = dataset.KITTI2015Dataset()

  dataset_loader = DataLoader(dataset=kitti2015,
                              batch_size=16,
                              shuffle=False)

  model = network.Network()
  print(dataset_loader)

  total_loss = 0
  correct = 0
  for i, (images, labels) in enumerate(dataset_loader):
    left_images  = Variable(images[0])
    right_images = Variable(images[1])
    labels = Variable(labels)

    outputs = model( (left_images, right_images) )

    criterion = nn.BCELoss()
    loss = criterion(outputs.data[0], labels)
    total_loss = total_loss + loss
    correct = correct + (0 if loss.item() > 0 else 1)

    if (i%200==199):
      print("["+str(i)+"] Loss:", total_loss.item()/2000, " | Correct:",correct,"/",i)

      total_loss = 0



if __name__ == '__main__':
  main()