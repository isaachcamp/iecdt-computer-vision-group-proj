import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader


class CloudDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, index_col=0)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels.columns)

    def __getitem__(self, idx: int):
        img_path = self.img_dir / f'{self.img_labels.columns[1]}.png'
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[:, idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def target_transform(target):
    """Convert target from pd.Series to 2D np.ndarray"""
    return target.values[:, np.newaxis]

def split_data(dataset, train=0.7, val=0.2):
    # Define dataset
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    train_size = int(np.round(train*total_size))
    val_size = int(np.round(val*total_size))
    test_size = total_size - train_size - val_size

    # Split indices
    train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

    # Define samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler)

    return train_loader, val_loader, test_loader