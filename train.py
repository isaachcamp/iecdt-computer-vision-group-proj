
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from define_dataset import CloudDataset, target_transform


IMG_PATH = Path('/gws/nopw/j04/iecdt/computer-vision-data/cam_a')
PATH = Path('/gws/nopw/j04/iecdt/JERMIT_the_frog/')


def train():    
    dataset = CloudDataset(
        PATH / 'hydrometeors_time_aligned_classes.csv', 
        IMG_PATH / 'compressed_rectified_imgs', 
        transform=ToTensor(),
        target_transform=target_transform
    )

    training_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for images, labels in training_dataloader:
        print(images.shape, labels.shape)
        break