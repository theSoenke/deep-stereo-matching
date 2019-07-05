import argparse
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import StereoDataset
from model import Model
from utils import loss_function, pixel_accuracy

learning_rate = 1e-2
half_range = 100

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='data/training')
parser.add_argument("--preprocess", type=str, default='preprocess/debug_15')
parser.add_argument("--checkpoint", type=str, default='checkpoint.pkl')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


writer = SummaryWriter()
model = Model(3, half_range*2+1).to(device)
if os.path.exists(checkpoint):
    print("Loading checkpoint")
    model.load_state_dict(torch.load(checkpoint))

optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.0005)
scheduler = MultiStepLR(optimizer, [24000, 32000, 40000], gamma=0.2)
train_dataset = StereoDataset(
    util_root=args.preprocess,
    data_root=args.data,
    filename='tr_160_18_100.bin',
    start_sample=0,
    num_samples=40000 * 128
)
val_dataset = StereoDataset(
    util_root=args.preprocess,
    data_root=args.data,
    filename='val_40_18_100.bin',
    start_sample=0,
    num_samples=1280,
)

train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_data = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4)
class_weights = torch.Tensor([1, 4, 10, 4, 1]).to(device)
samples = len(train_dataset)


def train(epoch):
    losses, accuracies, batch_times = np.array([]), np.array([]), np.array([])
    step = epoch * samples
    for batch in train_data:
        start_time = time.time()
        step += 1

        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target = batch['target'].to(device)

        _, _, pred = model(left_img, right_img)
        loss = loss_function(pred, target, class_weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        acc = pixel_accuracy(pred, target, pixel=2)
        losses = np.append(losses, loss.item())
        accuracies = np.append(accuracies, acc)
        batch_times = np.append(batch_times, time.time() - start_time)

        writer.add_scalar("train_loss", loss, global_step=step)
        writer.add_scalar("train_acc", acc, global_step=step)
        writer.add_scalar("learning_rate", scheduler.get_lr()[0], global_step=step)

        if step % 50 == 0:
            epoch_samples = (batch_size * (step // (epoch + 1)))
            mean_time = np.mean(batch_times) * 1000
            print("%d/%d samples, train_acc: %f, train_loss: %f, Time per batch: %fms" % (epoch_samples, samples, np.mean(accuracies), np.mean(losses), mean_time))
            losses, accuracies, batch_times = np.array([]), np.array([]), np.array([])

        if step % 500 == 0:
            torch.save(model.state_dict(), args.checkpoint)
            print("Created checkpoint")


@torch.no_grad()
def evaluate(epoch):
    losses, accuracies = np.array([]), np.array([])
    for batch in val_data:
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target = batch['target'].to(device)

        _, _, pred = model(left_img, right_img)
        loss = loss_function(pred, target, class_weights)

        acc = pixel_accuracy(pred, target, pixel=2)
        losses = np.append(losses, loss.item())
        accuracies = np.append(accuracies, acc)

    step = (epoch + 1) * samples
    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    writer.add_scalar("val_loss", avg_loss, global_step=step)
    writer.add_scalar("val_acc", avg_acc, global_step=step)
    print("Evaluation: val_acc: %f, val_loss: %f" % (avg_acc, avg_loss))


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters: %d" % trainable_params)
for epoch in range(epochs):
    model.train()
    train(epoch)

    model.eval()
    evaluate(epoch)

    print("Finished epoch %d" % epoch)

torch.save(model.state_dict(), args.checkpoint)
