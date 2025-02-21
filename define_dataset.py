import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset

class CloudDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, index_col=0)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

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
