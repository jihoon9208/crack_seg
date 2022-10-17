import warnings
warnings.filterwarnings('ignore')
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision
from trainer import Trainer
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from data_loaders import set_dataPath, CrackDataset
from utils import EarlyStopping

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Currently using "{device}" device.')

def get_model(output_channels=1, unfreeze=True):
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False)
    
    for param in model.parameters():
        param.requires_grad = unfreeze
    
    model.classifier = DeepLabHead(2048, output_channels)
    
    return model.to(device)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", default="mini_sample_crack_dataset/mini_crack_segmentation_dataset/")
    parser.add_argument("--save_path", default="seg_model.pth")

    parser.add_argument("--train_phase", type=str, default="train")
    parser.add_argument("--val_phase", type=str, default="val")
    parser.add_argument("--test_phase", type=str, default="test")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", default="1")
    parser.add_argument("--max_epoch", type=int, default=30)


    args = parser.parse_args()
    return args



def main(args):

    train = set_dataPath(args.dataset_path, args.train_phase)
    test = set_dataPath(args.dataset_path, args.test_phase)
    val = set_dataPath(args.dataset_path, args.val_phase)

    ## changed to define test size as % of dataset instead of set value
    # test_size = 0.30* data size

    print(f'Train size: {len(train)}, validation size: {len(val)} and test size: {len(test)}')

    train_gen = CrackDataset(args, train, device, True)
    val_gen = CrackDataset(args, val, device)
    #test_gen = CrackDataset(args, test, device)

    train_dataloader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True, collate_fn=train_gen.collate_fn, drop_last=True)
    valid_dataloader = DataLoader(val_gen, batch_size=1, shuffle=False, collate_fn=val_gen.collate_fn, drop_last=True)
    #test_dataloader = DataLoader(test_gen, batch_size=1, shuffle=False, collate_fn=test_gen.collate_fn, drop_last=True)


    model = get_model()
    early = EarlyStopping()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, min_lr=1e-6, factor=0.1)

    trainer = Trainer(
        config=args,
        data_loader=train_dataloader,
        val_data_loader=valid_dataloader,
        test_data_loader=test,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        function = early
    )

    trainer.train()


if __name__== "__main__":
    args = parse_args()

    main(args)