import pandas as pd
import numpy as np
import os

from PIL import Image

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


# Dataset
class CloudDataset(Dataset):
    '''
    Dataset to process the satellite images and segmentation masks

    Assigns:
    filenames (str): array of image filenames
    X (torch.FloatTensor): array of images
    mask (Dict): dictionary of masks for each class: fish, flower, gravel, sugar

    '''
    def __init__(self, split='train', cloud_types=['Fish', 'Flower', 'Gravel', 'Sugar']):
        '''
        Arguments:
        split (str): train, valid, or test set. Defaults to train.
        '''
        super().__init__()

        if split == 'train' or split == 'valid':
            path_to_data = './data/train_images'
        elif split == 'test':
            path_to_data = './data/test_images'
        else:
            raise ValueError(split)

        self.cloud_types = cloud_types

        # split train/val externally
        self.labels = pd.read_csv(f'./data/split_{split}.csv', dtype='object')
        self.img_files = [os.path.join(path_to_data, ff) for ff in self.labels['filename']]

        # define data augmentation
        self.transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),]
        )

    def __len__(self):
        '''Return length of the dataset'''
        return len(self.img_files)

    def __getitem__(self, idx):
        '''Return item at idx'''
        img_file = self.img_files[idx]
        img = Image.open(img_file)
        img_tensor = self.transforms(img)

        mask = {}

        for cloud in list(self.cloud_types):
            img_basefile = os.path.basename(img_file)
            enc = self.labels.query('filename == @img_basefile and class_name == @cloud')['EncodedPixels'].values[0]
            mask[cloud] = self.rle_to_mask(enc, height=1400, width=2100)

        mask_tensor = torch.stack(list(mask.values())).squeeze()

        return img_tensor, mask_tensor.type(torch.LongTensor)

    # Utility functions
    # from https://www.kaggle.com/robertkag/rle-to-mask-converter
    @staticmethod
    def rle_to_mask(rle_string,height,width):
        '''
        convert RLE(run length encoding) string to numpy array

        Parameters: 
        rleString (str): Description of arg1 
        height (int): height of the mask
        width (int): width of the mask 

        Returns: 
        torch.Tensor: torch tensor of the mask
        '''
        rows, cols = height, width
        if rle_string == '-1':
            return torch.zeros((height, width))
        else:
            rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
            rlePairs = np.array(rleNumbers).reshape(-1,2)
            img = np.zeros(rows*cols,dtype=np.uint8)
            for index,length in rlePairs:
                index -= 1
                img[index:index+length] = 255
            img = img.reshape(cols,rows)
            img = img.T
            return torch.from_numpy(img)

# Datamodule
class CloudDataModule(L.LightningDataModule):
    '''
    Data module for the cloud segmentation task

    Assigns
    cloud_types (List[str]): list of cloud types that are predicted
    '''
    def __init__(self, 
                 cloud_types=['Fish', 'Flower', 'Gravel', 'Sugar'],
                 batch_size=8,
                 num_workers=8
                 ):
        super().__init__()

        self.cloud_types = cloud_types
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self):
        self.train_data = CloudDataset(split='train', cloud_types=self.cloud_types)
        self.valid_data = CloudDataset(split='valid', cloud_types=self.cloud_types)
        self.test_data = CloudDataset(split='test', cloud_types=self.cloud_types)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

# Model architecture
class DummyModel(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)

        self.model = self.conv

    def forward(self, img):
        mask = self.conv(img)
        return mask

    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.forward(img)
        dice_loss = self.dice_loss(mask, pred_mask)
        return dice_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.forward(img)
        dice_loss = self.dice_loss(mask, pred_mask)

        self.log("valid_loss", dice_loss, on_epoch=True, sync_dist=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())


    @staticmethod
    def dice_loss(pred, target, smooth = 1.):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))