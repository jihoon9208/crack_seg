import cv2
import torch
import pandas as pd
from glob import glob


import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import random_crop, Rotate
import numpy as np

from torch.utils.data import Dataset

def set_dataPath(data_path, phase_name):

    assert phase_name in ['train', 'val', 'test']

    path_images = data_path + phase_name + '/input/'
    path_masks =  data_path + phase_name + '/output/'

    images_paths = glob(path_images + '*.jpg')
    masks_paths = glob(path_masks + '*.jpg')

    images_paths = sorted([str(p) for p in images_paths])
    masks_paths = sorted([str(p) for p in masks_paths])

    image_df = pd.DataFrame({'images': images_paths, 'masks': masks_paths})

    return image_df


class CrackDataset(Dataset):

    def __init__(self, args, dataset, device, augment=False):
        self.dataset = dataset.reset_index(drop=True)
        self.args = args
        self.device = device
        self.augment=augment
        self.batch_size = args.batch_size
        self.augmentations = A.Compose([
            A.Resize(self.args.image_size, self.args.image_size),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
            ToTensorV2()
        ])

        self.augmentations_val = A.Compose([
            A.Resize(self.args.image_size, self.args.image_size),
            ToTensorV2()
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.loc[idx].squeeze()
        image_path = row['images']
        mask_path = row['masks']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (self.args.image_size, self.args.image_size))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        if self.augment is True:
            image = random_crop(image, self.args.image_size, self.args.image_size )
            image = Rotate(image, angle=90)
            augmented = self.augmentations(image=image)
            image = augmented['image'].to(self.device, dtype=torch.float32)
        else :
            augmented = self.augmentations_val(image=image)
            image = augmented['image'].to(self.device, dtype=torch.float32)
        
        mask = torch.as_tensor(mask[None], dtype=torch.float32)
       
        return image, mask
    
    def collate_fn(self, batch):

        images, masks = tuple(zip(*batch))
        images = [img[None] for img in images]
        masks = [msk[None] for msk in masks]
        images, masks = [torch.cat(i).to(self.device) for i in [images, masks]]

        return images, masks

