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
            mask[cloud] = self.rle_to_mask(enc, height=1400, width=2100)

        mask_tensor = torch.stack(list(mask.values())).squeeze()
        if self.downscale:
            img_tensor = self.downscale_trafo(img_tensor.unsqueeze(0)).squeeze()
            mask_tensor = self.downscale_trafo(mask_tensor.unsqueeze(0)).squeeze()

        mask_tensor = mask_tensor.type(torch.LongTensor)
        if len(self.cloud_types) == 1:
            mask_tensor = F.one_hot(mask_tensor, num_classes=2)
            mask_tensor = mask_tensor.permute(2, 0, 1)

        return img_tensor, mask_tensor

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
                img[index:index+length] = 1
            img = img.reshape(cols,rows)
            img = img.T
            return torch.from_numpy(img)

    @staticmethod
    def mask_to_rle(img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        img = img.cpu().numpy()
        pixels= np.round(img.T.flatten()).astype(int)
        if np.all(np.allclose(pixels, 0)):
            return '-1' # for compatibility
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        
        if len(runs) % 2 == 1:
            runs = runs[:-1]
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

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


# Model architecture
class DummyModel(L.LightningModule):
    def __init__(self, num_masks=4):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=num_masks, kernel_size=3, padding=1)

        self.model = nn.Sequential(self.conv)

    def forward(self, img):
        mask = self.model(img)
        mask = F.sigmoid(mask)
        return mask

    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.forward(img)
        dice_loss = self.dice_loss(pred_mask, mask)
        return dice_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.forward(img)
        dice_loss = self.dice_loss(pred_mask, mask)

        self.log("valid_loss", dice_loss, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.forward(img)

        enc_pred_mask = CloudDataset.mask_to_rle(pred_mask.cpu().numpy())
        return enc_pred_mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())


    @staticmethod
    def dice_loss(pred, target, smooth = 1.):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

class CloudUNet(L.LightningModule):
    '''
    U-Net for cloud segmentation
    '''
    def __init__(self, num_classes=2):
        super().__init__()

        # architecture blocks
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # contracting
        self.down_conv1 = self.double_conv_block(3, 64)
        self.down_conv2 = self.double_conv_block(64, 128)
        self.down_conv3 = self.double_conv_block(128, 256)
        self.down_conv4 = self.double_conv_block(256, 512)
        self.down_conv5 = self.double_conv_block(512, 1024)

        # expanding
        self.up_transp5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv5 = self.double_conv_block(1024, 512)
        self.up_transp4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv4 = self.double_conv_block(512, 256)
        self.up_transp3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = self.double_conv_block(256, 128)
        self.up_transp2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = self.double_conv_block(128, 64)

        # output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, img):
        # contracting path
        down1 = self.down_conv1(img)
        x = self.max_pool(down1)
        down2 = self.down_conv2(x)
        x = self.max_pool(down2)
        down3 = self.down_conv3(x)
        x = self.max_pool(down3)
        down4 = self.down_conv4(x)
        x = self.max_pool(down4)
        down5 = self.down_conv5(x)

        # expanding path
        x = self.up_transp5(down5)
        down4 = down4[:, :, :x.shape[2], :x.shape[3]]
        x = self.up_conv5(torch.cat((down4, x), dim=1))

        x = self.up_transp4(x)
        down3 = down3[:, :, :x.shape[2], :x.shape[3]]
        x = self.up_conv4(torch.cat((down3, x), dim=1))

        x = self.up_transp3(x)
        down2 = down2[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat((down2, x), dim=1)
        x = self.up_conv3(x)

        x = self.up_transp2(x)
        down1 = down1[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat((down1, x), dim=1)
        x = self.up_conv2(x)

        return self.out(x)
    
    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.forward(img)
        dice_loss = self.dice_loss(pred_mask, mask)
        return dice_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.forward(img)
        dice_loss = self.dice_loss(pred_mask, mask)

        self.log("valid_loss", dice_loss, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.forward(img)

        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask)

        enc_pred_mask = CloudDataset.mask_to_rle(pred_mask.cpu().numpy())
        return enc_pred_mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @staticmethod
    def double_conv_block(in_channels, out_channels):
        '''
        Creates a double convolution block in UNet

        Arguments:
        in_channels (int): number of input channels
        out_channels (int): number of output channels

        Returns:
        torch.nn.Sequential
        '''
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        return block

    @staticmethod
    def dice_loss(pred, target, smooth = 1.):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))
