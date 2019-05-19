# deep-stereo-matching

## Preprocess data
1. Matlab or Octave needs to be installed
2. Replace path to dataset in preprocess/preprocess.m
3. cd preprocess
4. octave preprocess.m


## Train

    python3 train.py --data /data/kitti2015/training

## Inference

    python3 inference --data /data/kitti2015/testing --img_num 0
