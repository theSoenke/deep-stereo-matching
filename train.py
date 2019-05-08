import argparse
import os

from keras.layers import BatchNormalization, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam

from data_handler import DataHandler

epochs = 10
learning_rate = 1e-3
batch_size = 64


parser = argparse.ArgumentParser()
parser.add_argument('--data_root')
args = parser.parse_args()


def build_network():
    model = Sequential()
    for i in range(9):
        model.add(Conv2D(64, 3, activation='relu'))
        model.add(BatchNormalization())

    return model


data_loader = DataHandler(
    batch_size=batch_size,
    data_version='kitti2015',
    util_root='./preprocess/debug_15/',
    data_root=args.data_root,
    filename='tr_160_18_100.bin',
    num_tr_img=160,
    num_val_img=40,
)
data_loader.load()
