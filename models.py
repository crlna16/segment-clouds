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

from cloud_data import CloudDataset

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

class UNet(L.LightningModule):
    '''
    Generic U-Net for segmentation
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
