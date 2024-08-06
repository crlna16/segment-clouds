import sys
import time
import pandas as pd

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics.segmentation import MeanIoU # type: ignore

from lightning_modules import CloudDataModule, CloudDataset, DummyModel, CloudUNet

def main(args):
    print('Starting script')
    torch.manual_seed(404)

    # cloud_types = ['Fish', 'Flower', 'Gravel', 'Sugar']
    cloud_types = ['Fish']
    batch_size = 2
    downscale = True

    num_masks = len(cloud_types)

    cdm = CloudDataModule(cloud_types=cloud_types, batch_size=batch_size, downscale=downscale)
    cdm.setup()

    train_dataloader = cdm.train_dataloader()
    valid_dataloader = cdm.valid_dataloader()
    test_dataloader = cdm.test_dataloader()
    print('Finished data setup')

    #model = DummyModel(num_masks=num_masks)
    model = CloudUNet()
    print('Finished model setup')

    trainer = L.Trainer(max_epochs=50, 
                        enable_progress_bar=False,
                        fast_dev_run=False,
                        devices='auto',
                        callbacks=EarlyStopping('valid_loss', mode='min', patience=3), 
                        num_sanity_val_steps=1,)
    print('Starting training')
    trainer.fit(model=model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=valid_dataloader)
    print('Finished training')

    print('Generating predictions for holdout set')
    metrics = []
    encs = []
    idx = []
    metric = MeanIoU(num_classes=3)
    for i, (img, mask) in enumerate(test_dataloader):
        #pred_mask = torch.round(model(img)).squeeze(dim=1).type(torch.LongTensor)
        pred_mask = model(img)
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1).type(torch.LongTensor)

        #mask = torch.argmax(F.softmax(mask, dim=1), dim=1).type(torch.LongTensor)
        mask = mask.sum(dim=1)

        metrics.append(metric(pred_mask, mask).cpu().numpy())
        enc = CloudDataset.mask_to_rle(pred_mask)

        idx.append(i)
        encs.append(enc)

    df_pred = pd.DataFrame(dict(idx=idx, encoding=encs, mean_iou=metrics))
    df_pred.to_csv('./prediction/prediction.csv', index=False)
    
if __name__=='__main__':
    main(sys.argv[1:])