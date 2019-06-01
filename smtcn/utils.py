import matplotlib.pyplot as plt

def show_images(lefts, rights):
  size = 16
  fig = plt.figure(figsize=(size, size))

  rows = 1
  cols = 2

  # for i in range(1, rows):
  #   index = i*cols
  fig.add_subplot(rows, cols, 1)
  plt.imshow(lefts[0])

  fig.add_subplot(rows, cols, 2)
  plt.imshow(rights[0])

  plt.show()



