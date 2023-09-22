import random
import torch
import pandas as pd
from pathlib import Path
import numpy as np

import torch.utils.data as data
from torch.utils.data import dataloader


class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_path = self.dataset_cfg.csv_path

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle

        #---->split dataset
        if state == 'train':
            folds = [i for i in range(1, self.nfolds+1)]
            folds.remove(self.fold)
        if state == 'val':
            folds = [self.fold]
        if state == 'test':
            folds = [0]

        self.data = []
        self.label = []
        with open(self.csv_path, "r") as f:
            for row in f.readlines():
                name, label, fold, path = row.strip().split(",")
                if int(fold) in folds:
                    self.data.append(name)
                    self.label.append(int(label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        # features = torch.load(full_path)
        full_path = Path(self.feature_dir) / f'{slide_id}.npz'
        features = np.load(full_path)["features"]

        #----> shuffle
        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]


        return features, label

