import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

INPUT_DIR = 'datasets/KITTI2015/training'

# Images
img_left = mpimg.imread(INPUT_DIR + '/image_2/000000_10.png')
img_right = mpimg.imread(INPUT_DIR + '/image_3/000000_10.png')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
ax1.imshow(img_left)
ax2.imshow(img_right)
plt.show()