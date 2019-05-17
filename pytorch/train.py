import numpy as np
import torch
from torch import nn, optim
from torch.functional import F
from torch.utils.data import DataLoader

from data import StereoDataset
from model import Model

learning_rate = 1e-2
epochs = 10
batch_size = 64
half_range = 100
checkpoint = './checkpoint.pkl'


torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def three_pixel_loss(pred, target, weights):
    error = 0
    for i in range(pred.size(0)):
        pred_compare = pred[i, target[i][0]-2:target[i][0]+2+1]
        loss = torch.mul(pred_compare, weights).sum()
        error = error - loss

    return error

model = Model(3, half_range*2+1).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
dataset = StereoDataset(
    util_root='../preprocess/debug_15/',
    data_root='../data/training',
    filename='tr_160_18_100.bin',
)

train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
class_weights = torch.Tensor([1, 4, 10, 4, 1]).to(device)
samples = len(dataset)

for epoch in range(epochs):
    model.train()
    targets = np.tile(half_range, (batch_size, 1))
    target_batch = torch.tensor(targets, dtype=torch.int32)
    i = 0
    for batch in train_data:
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target = target_batch.to(device)

        _, _, pred = model(left_img, right_img)
        loss = three_pixel_loss(pred, target, class_weights)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        i += 1
        if i % 50 == 0:
            print("Loss %f" % loss)
            print("%d/%d samples" % ((batch_size * i), samples))

        if i % 500 == 0:
            torch.save(model.state_dict(), checkpoint)
            print("Created checkpoint")


    print("Finished epoch %d" % epoch)
    torch.save(model.state_dict(), checkpoint)
