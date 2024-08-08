import pandas as pd
import numpy as np
import os

from PIL import Image

import lightning as L
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

import utils

# Dataset
class CloudDataset(Dataset):
    '''
    Dataset to process the satellite images and segmentation masks

    Assigns:
    filenames (str): array of image filenames
    X (torch.FloatTensor): array of images
    mask (Dict): dictionary of masks for each class: fish, flower, gravel, sugar

    '''
    def __init__(self, split='train', cloud_types=['Fish', 'Flower', 'Gravel', 'Sugar'], downscale=False):
        '''
        Arguments:
        split (str): train, valid, or test set. Defaults to train.
        cloud_types (List): which cloud types to use for segmentation
        downscale (bool): If True, downscale images for faster training (default: False)
        '''
        super().__init__()

        if split in ['train', 'valid', 'dev']:
            path_to_data = './data/train_images'
        elif split == 'test':
            path_to_data = './data/test_images'
        else:
            raise ValueError(split)

        self.cloud_types = cloud_types

        # split train/val externally
        self.labels = pd.read_csv(f'./data/split_{split}.csv', dtype='object')

        # for single-mask models, remove samples that do not contain them
        if split != 'test' and len(self.cloud_types) == 1:
            self.labels = self.labels[self.labels['class_name'] == self.cloud_types[0]]
            self.labels = self.labels[self.labels['EncodedPixels'] != '-1']

        print(f'Samples in {split} after filter: {len(self.labels)}')

        self.img_files = [os.path.join(path_to_data, ff) for ff in self.labels['filename']]

        # define data augmentation
        self.transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),]
        )

        # define downscale operation
        self.downscale = downscale
        if self.downscale:
            #self.downscale_trafo = v2.Resize((350, 525))
            self.downscale_trafo = v2.Resize((1392, 2096))

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
            mask[cloud] = utils.rle_to_mask(enc, height=1400, width=2100)

        mask_tensor = torch.stack(list(mask.values())).squeeze()
        if self.downscale:
            img_tensor = self.downscale_trafo(img_tensor.unsqueeze(0)).squeeze()
            mask_tensor = self.downscale_trafo(mask_tensor.unsqueeze(0)).squeeze()

        mask_tensor = mask_tensor.type(torch.LongTensor)
        if len(self.cloud_types) == 1:
            mask_tensor = F.one_hot(mask_tensor, num_classes=2)
            mask_tensor = mask_tensor.permute(2, 0, 1)

        return img_tensor, mask_tensor

# Datamodule
class CloudDataModule(L.LightningDataModule):
    '''
    Data module for the cloud segmentation task

    Assigns
    cloud_types (List[str]): list of cloud types that are predicted
    is_dev_mode (bool): If True, predict on dev set. Else generate test set submission
    batch_size (int): batch size
    downscale (bool): Passed to Dataset
    num_workers (int): dataloader workers
    '''
    def __init__(self, 
                 cloud_types=['Fish', 'Flower', 'Gravel', 'Sugar'],
                 is_dev_mode=True,
                 batch_size=4,
                 downscale=False,
                 num_workers=8
                 ):
        super().__init__()

        self.cloud_types = cloud_types
        self.is_dev_mode = is_dev_mode
        self.batch_size = batch_size
        self.downscale = downscale
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self):
        self.train_data = CloudDataset(split='train', cloud_types=self.cloud_types, downscale=self.downscale)
        self.valid_data = CloudDataset(split='valid', cloud_types=self.cloud_types, downscale=self.downscale)
        self.dev_data = CloudDataset(split='dev', cloud_types=self.cloud_types, downscale=self.downscale)
        self.test_data = CloudDataset(split='test', cloud_types=self.cloud_types, downscale=self.downscale)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        if self.is_dev_mode:
            return DataLoader(self.dev_data, batch_size=1, num_workers=self.num_workers, shuffle=False)
        else:
            return DataLoader(self.test_data, batch_size=1, num_workers=self.num_workers, shuffle=False)