import argparse
import os

from keras.optimizers import Adam

from data_handler import DataHandler
from model import build_model

epochs = 10
learning_rate = 1e-3
batch_size = 64


parser = argparse.ArgumentParser()
parser.add_argument('--data_root')
args = parser.parse_args()


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

model = build_model(left_input_shape=data_loader.l_psz, right_input_shape=data_loader.r_psz)
train_samples = data_loader.pixel_loc.shape[0]

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
model.fit_generator(
    generator=data_loader.generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=10,
)
