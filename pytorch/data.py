import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from scipy.misc import imread
from torch.utils.data import Dataset


class StereoDataset(Dataset):
    def __init__(self, util_root, data_root, filename, num_val_loc=100, max_samples=-1):
        self.util_root = util_root
        self.data_root = data_root
        self.filename = filename
        self.num_val_loc = num_val_loc
        self.max_samples = max_samples

        fn = filename.split("_")
        fn[-1] = fn[-1].split(".")[0]
        data_split, num_images, psz, half_range = [fn[0]] + [int(x) for x in fn[1:]]
        self.num_images = num_images
        self.data_split = data_split
        self.half_range = half_range
        self.psz = psz

        print("Loading dataset..")
        self.load()


    def __len__(self):
        return self.pixel_loc.shape[0]

    def __getitem__(self, idx):
        sample_specs = tuple(self.pixel_loc[idx])
        img_id = sample_specs[0]
        img_left, img_right = self.ldata[img_id], self.rdata[img_id]

        l_patch = self.extract_patch(img_left, sample_specs, 0, "left")
        l_patch = torch.from_numpy(l_patch)
        l_patch = l_patch.permute(2, 0, 1)

        r_patch = self.extract_patch(img_right, sample_specs, self.half_range, "right")
        r_patch = torch.from_numpy(r_patch)
        r_patch = r_patch.permute(2, 0, 1)

        return {'left': l_patch, 'right': r_patch}


    def load(self, shuffle=True, bin_filename="myPerm.bin"):
        bin_path = os.path.join(self.util_root, bin_filename)
        self.file_ids = np.fromfile(bin_path, '<f4').astype(int)
        data_path = os.path.join(self.util_root, self.filename)
        self.pixel_loc = np.fromfile(data_path, '<f4').reshape(-1, 5).astype(np.int)
        self.pixel_loc[:, 2:5] = self.pixel_loc[:, 2:5] - 1
        self.pixel_loc = self.pixel_loc[:self.max_samples]
        if shuffle:
            np.random.shuffle(self.pixel_loc)

        self.ldata, self.rdata = {}, {}
        # TODO only load train or val samples
        for idx in range(len(self.file_ids)):
            fn = self.file_ids[idx]
            self.ldata[fn], self.rdata[fn] = self.load_sample(fn)

        if self.data_split == "val":
            self.pixel_loc = self.pixel_loc[:self.num_val_loc]

    def preprocess_image(self, img):
        img -= img.mean(axis=(0, 1))
        img /= img.std(axis=(0, 1))
        return img

    def extract_patch(self, x, sample_specs, half_range, side):
        img_id, loc_type, center_x, center_y, right_center_x = sample_specs
        right_center_y = center_y

        if side == "right":
            center_y, center_x = right_center_y, right_center_x

        patch = x[center_y-self.psz: center_y+self.psz+1, center_x-self.psz-half_range: center_x+self.psz+half_range+1]
        return patch

    def load_sample(self, fn):
        l_path = os.path.join(self.data_root, "image_2", "%06d_10.png" % fn)
        r_path = os.path.join(self.data_root, "image_3", "%06d_10.png" % fn)

        l_img, r_img = imread(l_path), imread(r_path)
        l_img = self.preprocess_image(l_img.astype(np.float32))
        r_img = self.preprocess_image(r_img.astype(np.float32))

        return l_img, r_img
