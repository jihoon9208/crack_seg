import random
import cv2
import numpy as np
import torch

def random_crop( image, crop_height, crop_width, h_start = 0, w_start = 0, p = 0.5):
    if (random.random() < p) :
        height, width = image.shape[:2]
        
        x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)

        image = image[y1:y2, x1:x2]
    
    return image

def get_random_crop_coords( height, width, crop_height, crop_width, h_start, w_start):
    
    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width + 1) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2

def Rotate(image, angle, p=0.25):

    if (random.random() < p) :
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2 - 0.5, height / 2 - 0.5), angle, 1.0)

        image = cv2.warpAffine(image, matrix, (width, height))

    return image

def extract_masks(segm, cl, n_cl):
    h, w  = segm.shape[0], segm.shape[1]
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)
    print(n_cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def replace_with_dict(array, dic):

    key = np.array(list(dic.keys()))
    value = np.array(list(dic.values()))

    sidx = key.argsort()
    
    return value[sidx[np.searchsorted(key,array, sorter=sidx)]]


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        
    def __call__(self, val_loss, model=None, path = None):
        if self.best_loss - val_loss > self.min_delta:
            torch.save(model.state_dict(), path)
            print(f'Model saved to: {path}')
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True