import sys
import time

import torch
import lightning as L

from lightning_modules import CloudDataModule, DummyModel

def main(args):
    print('Starting script')
    torch.manual_seed(404)

    cdm = CloudDataModule()
    cdm.setup()

    train_dataloader = cdm.train_dataloader()
    valid_dataloader = cdm.valid_dataloader()
    test_dataloader = cdm.test_dataloader()
    print('Finished data setup')

    model = DummyModel()
    print('Finished model setup')



    trainer = L.Trainer(max_epochs=10, 
                        num_sanity_val_steps=2,
                        )
    print('Starting training')
    trainer.fit(model=model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=valid_dataloader)
    print('Finished training')

if __name__=='__main__':
    main(sys.argv[1:])