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


# arguments parser config
parser = argparse.ArgumentParser()

parser.add_argument('--test_csv_path', type=str, help='test csv file')
parser.add_argument('--gpus', type=int, default=1,  help='Number of gpus to use for inference')
parser.add_argument('--test_batch_size', type=int, default=16,  help='Test/Evaluation batch size')
parser.add_argument('--num_tta', type=int, default=3,  help='Number of Test time Augmentations (TTA)')
parser.add_argument('--img_size', type=int, default=224,  help='input image size')
parser.add_argument('--seed_value', type=int, default=2020,  help='Seed value for reproducibility')
parser.add_argument('--specs_images_path', type=str, help='Direcetory containing log spectrograms images')
parser.add_argument('--save_resulting_file_to', type=str, help='Directory to save predictions file')
parser.add_argument('--arch', type=str, help='Model architecture to load for inference')


def load_models(models_path, arch=None, lr=0.013182567385564073):

    models = []
    for i in range(n_folds):
    models.append( AudioClassifier(arch_name=arch, lr=args.lr) )
    models[i].to(device)
    try:
        models[i].load_from_checkpoint(os.path.join(models_path, f'ZINDI-GIZ-NLP-AGRI-KEYWORDS-{arch}-{i}-based.ckpt'))
    except:
        models[i].load_from_checkpoint(os.path.join(models_path, f'ZINDI-GIZ-NLP-AGRI-KEYWORDS-{arch}-{i}-based-v0.ckpt'))
    models[i].eval()
    return models



def predict(test_df, batch_size=16, n_folds=3, transforms=None, n_tta=3, device='cuda', models=None):
  test_ds = AudioDataset(df=df, task='test', transforms=transforms)
  testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

  predictions_labels = []
  predictions_proba = []

  out = None

  for data in tqdm(testloader):
    x = data['image'].to(device)

    for i in range(n_folds):
      if i == 0: out = models[i](x)
      else: out += models[i](x)

    out /= n_folds
    out = F.softmax(input=out, dim=1)
    out_labels = out.argmax(1)
    out_probas = out.detach().cpu().numpy()

    
    predictions_labels += out_labels.tolist()
    predictions_proba += out_probas.tolist()

  return predictions_labels ,predictions_proba





if __name__ == '__main__':
    args = parser.parse_args()

    _ = seed_everything(args.seed_value)
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
    train, n_folds = make_folds(n_folds=args.kfold, args=args, data=df)