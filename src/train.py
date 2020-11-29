import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, Dataloader
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np

import pandas as pd 
import random
from models import AudioClassifier

import albumentations as al

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers, seed_everything 
import argparse


# arguments parser config
parser = argparse.ArgumentParser()

parser.add_argument('--train_csv_path', type=str, help='Train csv file')
parser.add_argument('--gpus', type=int, default=1,  help='gpu num')
parser.add_argument('--kfold', type=int, default=5,  help='number of folds to use for cross validation')
parser.add_argument('--train_batch_size', type=int, default=16,  help='Training batch size')
parser.add_argument('--test_batch_size', type=int, default=16,  help='Test/Evaluation batch size')
parser.add_argument('--num_epochs', type=int, default=25,  help='Number of epochs for training')
parser.add_argument('--img_size', type=int, default=224,  help='input image size')
parser.add_argument('--specs_images_path', type=str, help='Direcetory containing log spectrograms images')
parser.add_argument('--save_models_to', type=str, help='Directory to save trained models to')




def make_folds(data:pd.DataFrame, n_folds = 10, target_col='label'):
  data['fold'] = 0

  fold = StratifiedKFold(n_splits = n_folds, random_state=seed)
  for i, (tr, vr) in enumerate(fold.split(data, data[target_col])):
    data.loc[vr, 'fold'] = i

  return data, n_folds


def run_fold(fold, train_df, args,size=(224, 224), arch='resnet18', pretrained=True,   path='MODELS/'):
  
  torch.cuda.empty_cache()

  fold_train = train_df[train_df.fold != fold].reset_index(drop=True)
  fold_val = train_df[train_df.fold == fold].reset_index(drop=True)

  train_ds = AudioDataset(fold_train, size=size)
  val_ds = AudioDataset(fold_val, size=size)

  trainloader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
  validloader = DataLoader(val_ds, batch_size=args.test_batch_size, shuffle=False)

  model = AudioClassifier(arch_name=arch, lr=args.lr, pretrained=pretrained)

  tb_logger = loggers.TensorBoardLogger(save_dir='./runs', name='ZINDI-GIZ-NLP-AGRI-KEYWORDS', version=fold)

  ckpt_callback = pl.callbacks.ModelCheckpoint(filename=f'ZINDI-GIZ-NLP-AGRI-KEYWORDS-{model.hparams.arch_name}-{fold}-based', 
                                               dirpath=path, 
                                               monitor='val_logLoss', 
                                               mode='min')
  
  trainer = Trainer(max_epochs=args.num_epochs, gpus=args.gpus, logger=tb_logger, callbacks=[ckpt_callback])

  trainer.fit(model, trainloader, validloader)

  return trainer.logged_metrics



if __name__=='__main__':

    args = parser.parse_args()
    # data augmentations
    data_transforms = {
        'train': al.Compose([
                al.Resize(args.img_size, args.img_size),
                al.Cutout(p=.6, max_h_size=15, max_w_size=10, num_holes=4),
                al.Rotate(limit=35, p=.04),
                al.Normalize((0.1307,), (0.3081,))
        ]),

        'test': al.Compose([
                al.Resize(args.img_size, args.img_size),
                al.Cutout(p=.6, max_h_size=15, max_w_size=10, num_holes=4),
                al.Normalize((0.1307,), (0.3081,))
        ])
    }

    df = pd.read_csv(args.train_csv_path)
    train, n_folds = make_folds(n_folds=args.kfold, data=df)
    
    # traiining loop
    best_fold = 0
    avg_log_loss = 0.0
    best_logloss = np.inf

    for fold in n_folds:
        metrics = run_fold(fold=fold, train_df=train_df, args=args ,size=(224, 224), arch='resnet18', pretrained=True,   path=args.save_models_to)

        if metrics['val_logLoss'] < best_logloss:
            best_logloss = metrics['val_logLoss']
            best_fold = fold
            avg_log_loss += metrics['val_logLoss']
        else:
            avg_log_loss += metrics['val_logLoss']

    print(f'[INFO] raining done ! Avg LogLoss : {avg_log_loss / n_folds}')
