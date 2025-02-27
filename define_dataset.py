import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class CloudDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, index_col=0)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # drop the heights column from the count
        return len(self.img_labels.columns) - 1

    def __getitem__(self, idx: int):
        img_path = self.img_dir / f'{self.img_labels.columns[idx]}.png'
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[:, idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def target_transform(target):
    """Convert target from pd.Series to torch tensor"""
    target_arr = torch.from_numpy(target.values)
    # need float if using binary cross entropy loss, change this if we move to multi-class
    return target_arr.to(torch.float)

def target_transform_one_hot(target):
    """Convert target from pd.Series to one hot encoding of torch tensor"""
    target_arr = torch.from_numpy(target.values)
    return one_hot(target_arr, 2)

def split_data(dataset, batch_size, train=0.7, val=0.2):
    # seed for reproducibility to ensure we're getting the same dataset across evaluations
    generator1 = torch.Generator().manual_seed(42)
    random_splits = random_split(dataset, [train, val, 1-train-val], generator=generator1)

    train_loader = DataLoader(random_splits[0], batch_size, shuffle=True)
    val_loader = DataLoader(random_splits[1], batch_size)
    test_loader = DataLoader(random_splits[2], batch_size)

    return train_loader, val_loader, test_loader