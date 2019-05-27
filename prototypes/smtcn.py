import torch
import torch.nn as nn
import torch.optim as optim

import imageio
import numpy as np
import numpy.random as rng

import matplotlib.pyplot as plt

training_path = "datasets/KITTI2015/training"
dataset_pos = 1
dataset_neg_low = 8
dataset_neg_high = 16

# UTILS
def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def show_images(images):
  fig=plt.figure()
  for i in range(0, len(images)):
    fig.add_subplot(len(images), 1, i+1)
    image = images[i]
    if len(image.shape) == 2:
      plt.imshow(images[i], cmap="gray")
    else:
      plt.imshow(images[i])
  plt.show()

def extract_patch(image, center, size):
  result = np.arange(size[0]*size[1]).reshape(size[0],size[1])
  xStart = int(center[0]-size[0]/2)
  yStart = int(center[1]-size[1]/2)
  print(xStart,"|",yStart)
  for x in range(size[0]):
    for y in range(size[1]):
      result[x][y] = image[xStart + x][yStart + y]
  return result

# TRAINING
def training_image(index):
  X_left = rgb2gray(imageio.imread(training_path + "/image_2/" + index + ".png"))
  X_right = rgb2gray(imageio.imread(training_path + "/image_3/" + index + ".png"))
  y = imageio.imread(training_path + "/disp_noc_0/" + index + ".png")
  y = y/256 # Normalize to 0-255
  return X_left, X_right, y

def prepare_training_data():
  index = "000000_10"
  chnls = 1                                # Channels: 1->Grayscale, 3->RGB
  patch_size=(16,16)                          # Dimension of training patches
  LEFT, RIGHT, D = training_image( index )    # The images
  im_width = LEFT.shape[0]                    # Image width
  im_height = LEFT.shape[1]                   # Image height
  min_x = patch_size[0]                       # min x to be able to extract a patch
  max_x = im_width-patch_size[0]              # max x            -,,-
  max_y = patch_size[1]                       # min y            -,,-
  max_y = im_height-patch_size[1]             # max y            -,,-

  x = 300
  y = 225
 
  print("D", D[y,x])

  # Positive example
  # TODO: In paper there is o_pos= 1  x-d+_opos
  Pd = D[y,x]
  PL = extract_patch(LEFT,  (y, x),   patch_size)
  PR = extract_patch(RIGHT, (y, x-Pd), patch_size)

  # Negative example
  offset = rng.uniform( dataset_neg_low, dataset_neg_high )
  # If o_neg would lie outside of images, mirror it
  if (offset < min_x or offset > max_x):
    offset = rng.uniform( -dataset_neg_low, -dataset_neg_high )

  Nd = D[y,x]
  NL = extract_patch(LEFT,  (y, x),   patch_size)
  NR = extract_patch(RIGHT, (y, x-Nd+offset), patch_size)

  #show_images()
  #return [PL,NL], [PR,NR]

  epoch = 0
  running_loss = 0.0
  ## MODEL
  model  = Network()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  left   = torch.FloatTensor(np.asarray(np.array([PL]).reshape(1, chnls, 16, 16)))
  right  = torch.FloatTensor(np.asarray(np.array([PR]).reshape(1, chnls, 16, 16)))
  output = model([left,right])
  loss   = criterion(output, Pd)

  running_loss += loss.item()
  #if i % 2000 == 1999:    # print every 2000 mini-batches
  print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
  running_loss = 0.0

  print(output)


class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.left = nn.Sequential( 
      nn.Conv2d(1, 20, kernel_size=4, stride=1, padding=1),
      nn.ReLU(),
      nn.LayerNorm([15,15]),
      nn.Conv2d(20, 20, kernel_size=4, stride=1, padding=1),
      nn.ReLU(),
    )
    self.right = nn.Sequential( 
      nn.Conv2d(1, 20, kernel_size=4, stride=1, padding=1),
      nn.ReLU(),
      nn.LayerNorm([15,15]),
      nn.Conv2d(20, 20, kernel_size=4, stride=1, padding=1),
      nn.ReLU(),
    )
  def forward(self, input):      
    return torch.cat((self.left(input[0]), self.right(input[1])))
    

def main():
  prepare_training_data()
  # X_left, X_right, y = training_image("000000_10")
  # show_images([X_left, X_right, y])

if __name__ == '__main__':
  main()