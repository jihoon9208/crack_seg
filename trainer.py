import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.autonotebook import tqdm

class Trainer:

    def __init__(
        self,
        config,
        data_loader,
        val_data_loader,
        test_data_loader,
        model,
        optimizer,
        scheduler ,
        criterion,
        function ):
        
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.early = function

        self.train_dataloader = data_loader
        self.valid_dataloader = val_data_loader
        self.test_dataloader = test_data_loader

        self.start_epoch = 1
        self.max_epoch = config.max_epoch
        self.save_path = config.save_path

        self.augmentations_test = A.Compose([
            A.Resize(self.config.image_size, self.config.image_size),
            ToTensorV2()
        ])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Currently using "{self.device}" device.')

    def train_one_batch(self, batch, model, criterion, optimizer):
        images, masks = batch
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output['out'], masks)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def validate_one_batch(self, batch, model, criterion):
        images, masks = batch
        output = model(images)
        loss = criterion(output['out'], masks)
        return loss.item()

    @torch.no_grad()
    def validate_test_image(self, model, dataset):
        idx = np.random.randint(len(dataset))
        dataset = dataset.reset_index(drop=True)
        row = dataset.loc[idx].squeeze()
        
        image = cv2.imread(row['images'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.augmentations_test(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device, dtype=torch.float32)
        
        mask = cv2.imread(row['masks'])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (self.config.image_size, self.config.image_size))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        model.eval()
        output = model(image_tensor)
        output = output['out'][0].cpu().detach().numpy().transpose(1,2,0)
        
        plt.figure(figsize=(8, 4))
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
        plt.pause(0.001)
    
    def train(self):
        train_losses, valid_losses = [], []

        for epoch in range(self.start_epoch,self.max_epoch+1):

            print(f'Epoch {epoch + 1}/{self.max_epoch}')
            epoch_train_losses, epoch_valid_losses = [], []
            
            self.model.train()
            for _, batch in enumerate(tqdm(self.train_dataloader, leave=False)):
                batch_train_loss = self.train_one_batch(batch, self.model, self.criterion, self.optimizer)
                epoch_train_losses.append(batch_train_loss)
            epoch_train_loss = np.array(epoch_train_losses).mean()
            train_losses.append(epoch_train_loss)

            print(f'Train loss: {epoch_train_loss:.4f}.')
            
            self.model.eval()
            for i, batch in enumerate(tqdm(self.valid_dataloader, leave=False)):
                batch_valid_loss = self.validate_one_batch(batch, self.model, self.criterion)
                epoch_valid_losses.append(batch_valid_loss)
            epoch_valid_loss = np.array(epoch_valid_losses).mean()
            valid_losses.append(epoch_valid_loss)
            print(f'Valid loss: {epoch_valid_loss:.4f}.')
            print('-'*50)    
            
            #self.validate_test_image(self.model, self.test_dataloader)
            self.scheduler.step(epoch_valid_loss)

            self.early(epoch_valid_loss, model=self.model, path = self.save_path)
            if self.early.early_stop:
                print(f'Validation loss did not improve for {self.early.patience} epochs. Training stopped.')
                self.model.load_state_dict(torch.load(self.save_path))
                break


    
                