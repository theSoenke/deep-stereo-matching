import imageio
import numpy as np
import numpy.random as rng
from torch.autograd import Variable

import utils

import torch
from torch.utils.data.dataset import Dataset

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def extract_patch(image, center, size):
  result = np.arange(size*size).reshape(size,size)
  xStart = int(center[0]-size/2)
  yStart = int(center[1]-size/2)
  for x in range(size):
    for y in range(size):
      result[x][y] = image[xStart + x][yStart + y]
  return result

class KITTI2015Dataset(Dataset):
  def __init__(self,path="datasets/KITTI2015"):
    self.training_path = path+"/training"
    self.testing_path = path+"/testing"
    self.patch_size = 32
    self.channels = 1
    self.neg_offset = 32
    self.seed = 129

    self.num_images = 5
    self.samplesPerImage = 16

    self.patches = list()
    self.labels = list()

    self.cache_image_id = -1
    self.cache_patches = dict()
    self.cache_labels = dict()

  
  def __getitem__(self, index):
    image_id = int(index / self.samplesPerImage)
    patch_id = int(index % self.samplesPerImage)


    # patches, labels, _ = self.prepare_image(str(image_id).zfill(6)+"_10")

    # return (patches[patch_id], labels[patch_id])

    if not image_id in self.cache_patches:
      print("prepare patches for image", image_id)
      patches, labels, _ = self.prepare_image(str(image_id).zfill(6)+"_10")
      self.cache_patches[image_id] = patches
      self.cache_labels[image_id] = labels

    return (self.cache_patches[image_id][patch_id],self.cache_labels[image_id][patch_id])


    # if (self.cache_image_id != image_id):
    #   #print("Generate patches for image ", image_id)
    #   self.cache_patches, self.cache_labels, self.cache_images = self.prepare_image(str(index).zfill(6)+"_10")
    #   self.cache_image_id = image_id

    # return self.cache_patches[patch_id], self.cache_labels[patch_id], self.cache_images

    # patches, label = self.prepare_image(str(index).zfill(6)+"_10")
    # return (patches, label)

  def __len__(self):
    return self.num_images * self.samplesPerImage #len(self.patches)

  def prepare_image(self, index):
    # load images
    X_left  = rgb2gray(imageio.imread(self.training_path + "/image_2/" + index + ".png"))
    X_right = rgb2gray(imageio.imread(self.training_path + "/image_3/" + index + ".png"))
    D = imageio.imread(self.training_path + "/disp_noc_0/" + index + ".png")
    D = D/256 # Normalize to 0-255

    width = X_left.shape[1]
    height = X_left.shape[0]

    # Helper vars to determine if a patch is inside the image
    min_x = self.patch_size
    max_x = width-self.patch_size
    min_y = self.patch_size
    max_y = height-self.patch_size

    patches = list()
    labels = list()
    images = list()

    rng.seed( self.seed )
    for i in range(int(self.samplesPerImage)):
      is_neg = (i%2 == 0) #is negative example

      # Calculate what pixel we want to use
      x = int(rng.uniform(min_x, max_x))
      y = int(rng.uniform(min_y, max_y))
      offset = 0
      if is_neg:
        offset = rng.uniform(self.neg_offset, self.neg_offset*2)

      # The ground truth disparity at the pixel
      d = D[y,x]

      # Disparity truth lies outside of image
      if (x+d > max_x):
        i = i-1
        continue
   
      if x+d-offset > max_x-1:
        offset = -offset

      #print(not is_neg, offset)

      # Extract patches from images
      Li = extract_patch(X_left,  (y, x),   self.patch_size)
      L = torch.from_numpy( Li ).type("torch.FloatTensor")
      L = L.reshape( 1, self.patch_size, self.patch_size )

      Ri = extract_patch(X_right, (y, x-d+offset), self.patch_size)
      R = torch.from_numpy( Ri ).type("torch.FloatTensor")
      R = R.reshape( 1, self.patch_size, self.patch_size )

      #utils.show_images([Li],[Ri])

      if is_neg:
        label = torch.FloatTensor([0.0])
      else:
        label = torch.FloatTensor([1.0])

      images.append((Li,Ri))
      patches.append((L,R))
      labels.append(label)

    return patches, labels, images
