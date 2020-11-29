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


class AudioClassifier(pl.LightningModule):
    def __init__(self, pretrained=True, out_size=193,img_size=224, lr=1e-4, arch_name='resnet34'):
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
                nn.Dropout(.3),
                nn.Linear(self.num_last_ftrs, out_size)
            ) 
            torch.nn.init.xavier_normal_(self.arch.fc[1].weight)

        else:
            self.arch = getattr(models, arch_name)(pretrained)

            head = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
            head.weight = torch.nn.Parameter(self.arch.conv1.weight.sum(dim=1, keepdim=True))

            self.arch.conv1 = head
            # classifier part
            self.num_last_ftrs = getattr(self.arch, 'fc').in_features
            self.arch.fc = nn.Sequential(
                nn.Dropout(.3),
                nn.Linear(self.num_last_ftrs, out_size)
            ) 
            torch.nn.init.xavier_normal_(self.arch.fc[1].weight)



    def forward(self, x):
        x = self.arch(x.view(-1, 1, self.hparams.img_size, self.hparams.img_size))

        return x


    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        return opt



    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)

        logLoss = self.get_loss(logits=logits.detach().cpu().numpy(), targets=y.detach().cpu().numpy())
        acc = self.get_acc(logits=logits.detach().cpu().numpy(), targets=y.detach().cpu().numpy())

        # logging 
        self.log('train_acc', acc, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_logLoss', logLoss, on_epoch=True, on_step=True, prog_bar=False)

        return {'loss':logLoss, 'train_logloss':logLoss, 'train_acc':acc}


    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)

        val_loss = self.get_loss(logits=logits.detach().cpu().numpy(), targets=y.detach().cpu().numpy())
        val_acc = self.get_acc(logits=logits.detach().cpu().numpy(), targets=y.detach().cpu().numpy())

        # logging 
        self.log('val_acc', val_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_logLoss', val_loss, on_epoch=True, on_step=False, prog_bar=True)

        return {'val_logLoss':val_loss, 'val_acc':val_acc}


    def get_acc(self, logits, targets):
        preds = F.softmax(logits, dim=1).argmax(1)
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
                al.Cutout(p=.6, max_h_size=15, max_w_size=10, num_holes=4),
                al.Rotate(limit=35, p=.04),
                al.Normalize((0.1307,), (0.3081,))
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
    model = AudioClassifier(arch_name='resnet34', pretrained=True).to('cuda')
    batch = next(iter(dl))
    logits = model(batch['image'].to('cuda'))
    print(logits.shape, batch['label'].shape)
    loss = model.get_loss(logits, batch['label'].to('cuda'))
    acc = model.get_acc(logits, batch['label'].to('cuda'))

    print(f"loss : {loss} | acc {acc}")
