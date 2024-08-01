import sys
import time
import pandas as pd

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics.segmentation import MeanIoU

from lightning_modules import CloudDataModule, CloudDataset, DummyModel

def main(args):
    print('Starting script')
    torch.manual_seed(404)

    # cloud_types = ['Fish', 'Flower', 'Gravel', 'Sugar']
    cloud_types = ['Fish']
    batch_size = 16

    num_masks = len(cloud_types)

    cdm = CloudDataModule(cloud_types=cloud_types, batch_size=batch_size)
    cdm.setup()

    train_dataloader = cdm.train_dataloader()
    valid_dataloader = cdm.valid_dataloader()
    test_dataloader = cdm.test_dataloader()
    print('Finished data setup')

    model = DummyModel(num_masks=num_masks)
    print('Finished model setup')

    trainer = L.Trainer(max_epochs=1, 
                        fast_dev_run=False,
                        callbacks=EarlyStopping('valid_loss', mode='min'), 
                        num_sanity_val_steps=2,)
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
        pred_mask = torch.round(model(img)).squeeze(dim=1).type(torch.int)
        metrics.append(metric(pred_mask, mask).cpu().numpy())
        enc = CloudDataset.mask_to_rle(pred_mask)

        idx.append(i)
        encs.append(enc)

    df_pred = pd.DataFrame(dict(idx=idx, encoding=encs, mean_iou=metrics))
    df_pred.to_csv('./prediction/prediction.csv', index=False)
    
if __name__=='__main__':
    main(sys.argv[1:])