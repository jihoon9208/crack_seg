import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from tqdm.autonotebook import tqdm

import os

from sklearn.model_selection import train_test_split

from glob import glob
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from crack_segmentation import get_model
from data_loaders import set_dataPath, CrackDataset

from utils import extract_both_masks, extract_classes, union_classes, replace_with_dict
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", default="mini_sample_crack_dataset/mini_crack_segmentation_dataset/")
    parser.add_argument("--save_path", default="seg_model.pth")

    parser.add_argument("--train_phase", type=str, default="train")
    parser.add_argument("--val_phase", type=str, default="val")
    parser.add_argument("--test_phase", type=str, default="test")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--max_epoch", type=int, default=30)

    args = parser.parse_args()
    return args

class Evaluate:

    def __init__(self, config, test_dataset):
        self.config = config
        self.test_dataset = test_dataset
        self.checkpoint = torch.load(self.config.save_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = config.image_size
        self.model = get_model()
        self.model.load_state_dict(self.checkpoint)

        self.augmentations_test = A.Compose([
            A.Resize(self.config.image_size, self.config.image_size),
            ToTensorV2()
        ])


    @torch.no_grad()
    def validate_test_image(self):
        
        dataset = self.test_dataset.reset_index(drop=True)

        for idx in range(len(self.test_dataset)):    
            row = dataset.loc[idx].squeeze()

            file, ext = os.path.splitext(row['images'])
            image_name = file.split('/')[-1]
            folder_name = 'out_' + image_name
            
            image = cv2.imread(row['images'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = self.augmentations_test(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(self.device, dtype=torch.float32)
            
            mask = cv2.imread(row['masks'])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (self.config.image_size, self.config.image_size))
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            self.model.eval()
            output = self.model(image_tensor)
            output = output['out'][0].cpu().detach().numpy().transpose(1,2,0)
            
            """ plt.figure(figsize=(8, 4))
            plt.subplot(131)
            plt.title('Original image')
            plt.imshow(image)
            
            plt.subplot(132)
            plt.title('Original mask')
            plt.imshow(mask, cmap='gray')
            
            plt.subplot(133)
            plt.title('Predicted mask')
            plt.imshow(output, cmap='gray')
            
            plt.tight_layout()
            plt.show()
            plt.pause(0.001) """

            filename = os.path.join('real_images','img_{}.png'.format(image_name))

            cv2.imwrite(filename, output)
            
    

    def mean_IU(self, eval_segm, gt_segm):
    
        '''
    
            n_cl : the number of classes
            t_i : the total number of pixels in class i

            n_ij : the number of pixels of class i predicted to belong to class j. So for class i:
            n_ii : the number of correctly classified pixels (true positives)
            n_ij : the number of pixels wrongly classified (false positives)
            n_ji : the number of pixels wrongly not classifed (false negatives)

        
            (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
        '''

        cl, n_cl   = union_classes(eval_segm, gt_segm)
        _, n_cl_gt = extract_classes(gt_segm)
        eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        IU = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]
    
            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i  = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

            IU[i] = n_ii / (t_i + n_ij - n_ii)

        mean_IU_ = np.sum(IU) / n_cl_gt

        return mean_IU_

    def if_wIoU(eval_segm, gt_segm):

        cl, n_cl   = union_classes(eval_segm, gt_segm)
        _, n_cl_gt = extract_classes(gt_segm)
        eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)


        IU = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]
    
            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i  = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

            

        mean_IU_ = np.sum(IU) / n_cl_gt

        return mean_IU_


    def predict_on_crops(self):

        idx = np.random.randint(len(self.test_dataset))
        test_dataset = self.test_dataset.reset_index(drop=True)
        row = test_dataset.loc[idx].squeeze()

        image = cv2.imread(row['images'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.augmentations_test(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device, dtype=torch.float32)
        
        tmp_image = augmented['image']
        tmp_image = tmp_image.detach().cpu().numpy().transpose(2,1,0).copy()
        
        height, width, channels = image.shape
        number_pic=0
        output_image = np.zeros_like(image)

        for i in range(0,height, int(self.image_size/4) ):

            for j in range(0,width,int(self.image_size/4 )):

                crop_img = image_tensor[:,:,i:i+ int(self.image_size/4) , j:j+int(self.image_size/4)]
                predicted_class = self.predict(crop_img)

                ## save image
                file, ext = os.path.splitext(row['images'])
                image_name = file.split('/')[-1]
                folder_name = 'out_' + image_name

                ## Put predicted class on the image
                if str(predicted_class) == 'Positive':
                    color = (0,0, 255)
                else:
                    color = (0, 255, 0)

                cv2.putText(tmp_image, str(predicted_class), ((int)(10), (int)(10)), cv2.FONT_HERSHEY_SIMPLEX , 0.2, color, 1, cv2.LINE_AA) 
                
                empty_image = np.zeros_like(tmp_image, dtype=np.uint8)
                empty_image[:] = color
                
                add_img = cv2.addWeighted(tmp_image, 0.9, empty_image, 0.1, 0)

                plt.imshow(add_img)
                plt.show()
                
                if not os.path.exists(os.path.join('real_images', folder_name)):
                    os.makedirs(os.path.join('real_images', folder_name))
                filename = os.path.join('real_images', folder_name,'img_{}.png'.format(number_pic))
                
                cv2.imwrite(filename, add_img)
                output_image[ i:i + self.image_size, j:j + self.image_size,: ] = add_img

                number_pic+=1

        ## Save output image
        cv2.imwrite(os.path.join('real_images','predictions', folder_name + '.png'), output_image)
        
        plt.subplot(131)
        plt.title('Original image')
        plt.imshow(output_image)
        plt.show()
        

    def predict(self, image_tensor):
        # it uses the model to predict on test_image...
        idx_to_class = {0:'Negative', 1:'Positive'}

        with torch.no_grad():
            self.model.eval()
            
            output = self.model(image_tensor)
            tmp_img = output['out'][0].cpu().detach().numpy().transpose(1,2,0)
            tmp_img = tmp_img.squeeze(-1)

            # this computes the probability of each classes.
            """ ps = torch.exp(output['out'][0])
            topk, topclass = ps.topk(1, dim=1)
            
            tmp = topclass.detach().cpu().numpy()[0][0] """

            tmp_img[tmp_img < 30] = 0
            tmp_img[tmp_img >= 30] = 1

            class_name = replace_with_dict(tmp_img, idx_to_class)

        return class_name

def main(args):

    test = set_dataPath(args.dataset_path, args.test_phase)
    print(f'test size: {len(test)}')

    evaluate = Evaluate(
        config = args,
        test_dataset= test
    )
    #evaluate.predict_on_crops()
    evaluate.validate_test_image()

if __name__== "__main__":
    args = parse_args()

    main(args)