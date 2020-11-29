import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import pandas as pd 
import random
import albumentations as al

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers, seed_everything 
from efficientnet_pytorch import EfficientNet 

from datasets import AudioDataset
import warnings
warnings.filterwarnings(action='ignore')






class AudioClassifier(pl.LightningModule):
    def __init__(self, pretrained=True, out_size=193,img_size=224, lr=0.0023182567385564073, arch_name='resnet34'):
        super(AudioClassifier, self).__init__()
        self.save_hyperparameters()

        if 'efficient' in self.hparams.arch_name:
            self.arch = Efficientnet.from_pretrained(self.hparams.arch_name)

            #change firs conv layer to accept grayscale images
            head = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
            head.weight = torch.nn.Parameter(self.arch.conv1.weight.sum(dim=1, keepdim=True))
            self.conv1 = head

            # add our own  classifier 
            self.num_last_ftrs = getattr(self.arch, 'fc').in_features
            self.arch.fc = nn.Sequential(
                nn.Dropout(.5),
                nn.Linear(self.num_last_ftrs, out_size)
            ) 
            torch.nn.init.xavier_normal_(self.arch.fc[1].weight)

        elif 'resnet' in self.hparams.arch_name:
            self.arch = getattr(models, arch_name)(pretrained)
            for param in self.arch.parameters():
                param.require_grad = False


            head = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
            head.weight = torch.nn.Parameter(self.arch.conv1.weight.sum(dim=1, keepdim=True))

            self.arch.conv1 = head
            # classifier part
            self.num_last_ftrs = getattr(self.arch, 'fc').in_features
            self.arch.fc = nn.Sequential(
                nn.Dropout(.5),
                nn.Linear(self.num_last_ftrs, out_size)
            ) 
            torch.nn.init.xavier_normal_(self.arch.fc[1].weight)

        else:
            self.arch = nn.Sequential(
                nn.Conv2d(1, 8, (5, 5)),
                nn.Conv2d(8, 16, (5, 5)),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.MaxPool2d((2, 2)),

                nn.Conv2d(16, 64, (5, 5)),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d((2, 2)),


                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d((3, 3)),

                nn.Conv2d(128, 256, (3, 3)),
                nn.Conv2d(256, 512, (3, 3)),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d((2, 2)),

            )
            self.classifier = nn.Sequential(
                nn.Linear(6 * 6 * 512, 256),
                nn.Linear(256, self.hparams.out_size)
            )

    def forward(self, x):
        x = self.arch(x.view(-1, 1, self.hparams.img_size, self.hparams.img_size))
        try:
            x = x.view(-1, 6 * 6 * 512)
            x = self.classifier(x)
        except :
            pass

        return x


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return opt



    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)

        logLoss = self.get_loss(logits=logits, targets=y)
        acc = self.get_acc(logits=logits, targets=y)

        # logging 
        self.log('train_acc', acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train_logLoss', logLoss, on_epoch=True, on_step=True, prog_bar=False)

        return {'loss':logLoss, 'train_logLoss':logLoss, 'train_acc':acc}


    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)

        val_loss = self.get_loss(logits=logits, targets=y)
        val_acc = self.get_acc(logits=logits, targets=y)

        # logging 
        self.log('val_acc', val_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_logLoss', val_loss, on_epoch=True, on_step=False, prog_bar=True)

        return {'val_logLoss':val_loss, 'val_acc':val_acc}


    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)

        test_loss = self.get_loss(logits=logits, targets=y)
        test_acc = self.get_acc(logits=logits, targets=y)

        # logging 
        self.log('test_acc', test_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_logLoss', test_loss, on_epoch=True, on_step=False, prog_bar=True)

        return {'test_logLoss':test_loss, 'test_acc':test_acc}



    def get_acc(self, logits, targets):

        preds = nn.functional.softmax(logits, dim=1).argmax(1)

        acc = (preds == targets).float().mean()
        return acc


    def get_loss(self, logits, targets):
        
        loss = nn.CrossEntropyLoss()(logits, targets)
        return loss



if __name__ == '__main__':
    IMG_SIZE = 224

    # data augmentations
    data_transforms = {
        'train': al.Compose([
                al.Resize(IMG_SIZE, IMG_SIZE),
 
        ]),

        'test': al.Compose([
                al.Resize(IMG_SIZE, IMG_SIZE),
                al.Cutout(p=.6, max_h_size=15, max_w_size=10, num_holes=4),
                al.Normalize((0.1307,), (0.3081,))
        ])
    }

    # data loaders
    train_df = pd.read_csv('../data/Giz-agri-keywords-data/final_train.csv')
    ds = AudioDataset(images_path='../data/Giz-agri-keywords-data/datasets/images', df=train_df, transforms=data_transforms['train'])
    dl = DataLoader(dataset=ds, shuffle=True, batch_size=32, num_workers=os.cpu_count())

    # instanciate model
    model = AudioClassifier(arch_name='scratch', pretrained=True).to('cuda')
    #batch = next(iter(dl))
    #logits = model(batch['image'].to('cuda'))
    #print(logits.shape)
    #print(logits.shape, batch['label'].shape)
    #loss = model.get_loss(logits, batch['label'].to('cuda'))
    #acc = model.get_acc(logits, batch['label'].to('cuda'))

    trainer = pl.Trainer(gpus=1, max_epochs=5)
    trainer.fit(model, dl)
    trainer.test(model, dl)
    #trainer.tune(model, train_dataloader=dl)
    print(trainer.logged_metrics)

