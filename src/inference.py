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
from models import AudioClassifier

import warnings
warnings.filterwarnings(action='ignore')
import argparse

from tqdm import tqdm






# arguments parser config
parser = argparse.ArgumentParser()

parser.add_argument('--test_csv_path', type=str, help='test csv file')
parser.add_argument('--sample_csv_path', type=str, help='sample submission csv file')
parser.add_argument('--gpus', type=int, default=1,  help='Number of gpus to use for inference')
parser.add_argument('--train_batch_size', type=int, default=64,  help='batch size used for training')
parser.add_argument('--test_batch_size', type=int, default=16,  help='Test/Evaluation batch size')
parser.add_argument('--n_tta', type=int, default=3,  help='Number of Test time Augmentations (TTA)')
parser.add_argument('--kfold', type=int, default=3,  help='Number of folds used for training')
parser.add_argument('--img_size', type=int, default=224,  help='input image size')
parser.add_argument('--seed_value', type=int, default=2020,  help='Seed value for reproducibility')
parser.add_argument('--specs_images_path', type=str, help='Direcetory containing log spectrograms images')
parser.add_argument('--save_resulting_file_to', type=str, help='Directory to save predictions file')
parser.add_argument('--arch', type=str, help='Model architecture to load for inference')
parser.add_argument('--num_epochs', type=int, default=40,  help='Number of epochs for training')
parser.add_argument('--models_path', type=str, help='Direcetory containing models checkpoints')
parser.add_argument('--lr', type=float, default=0.013182567385564073,  help='Learning rate for model training')




def load_models(models_path, arch=None, n_folds=3, device='cuda'):

    models = []
    for i in range(n_folds):
        models.append( AudioClassifier(arch_name=arch) )
        models[i].to(device)
        try:
            models[i].load_from_checkpoint(os.path.join(models_path, f'ZINDI-GIZ-NLP-AGRI-KEYWORDS-{arch}-{i}-based.ckpt'))
        except:
            models[i].load_from_checkpoint(os.path.join(models_path, f'ZINDI-GIZ-NLP-AGRI-KEYWORDS-{arch}-{i}-based-v0.ckpt'))
        models[i].eval()

    return models



def predict(test_df, images_path, batch_size=16, n_folds=3, transforms=None, n_tta=3, device='cuda', models=None):
    # create test AudioDataset
    test_ds = AudioDataset(images_path=images_path, task='test', df=test_df, transforms=transforms)
    test_dl = DataLoader(dataset=test_ds, shuffle=False, batch_size=batch_size)

    predictions_labels = []
    predictions_proba = []

    out = None

    for data in tqdm(test_dl):
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



def make_submission_file(sub:pd.DataFrame,predictions_proba=None, submissions_folder=None, params=None):
    submission = pd.DataFrame()
    words = sub.columns[2:]
    submission['fn'] = sub['fn']
    for i, label in enumerate(words):
        submission[label] = 0.
    for i, label in enumerate(words):
        submission.loc[:,label] = np.array(predictions_proba)[:,i]

    train_batch_size,_, n_folds, img_size, n_epochs, arch = params.values()

    csv_file = f'GIZ_SIZE_{img_size}_arch_{arch}_n_folds_{n_folds}_num_epochs_{n_epochs}_train_bs_{train_batch_size}.csv'
    submission.to_csv(os.path.join(submissions_folder, csv_file), index=False)

    print(f'[INFO] Submission file save to {os.path.join(submissions_folder, csv_file)}')

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

    test = pd.read_csv(args.test_csv_path)
    sample = pd.read_csv(args.sample_csv_path)

    # load models
    models = load_models(models_path=args.models_path, n_folds=args.kfold, arch=args.arch)
    # make predictions
    predictions_labels, predictions_proba = predict(test_df=test, 
                                                    images_path=args.specs_images_path,
                                                    batch_size=args.test_batch_size, 
                                                    n_folds=args.kfold, 
                                                    transforms=data_transforms['test'], 
                                                    n_tta=args.n_tta, 
                                                    device='cuda', 
                                                    models=models)

    params = {
        'train_batch_size': args.train_batch_size,
        'test_batch_size':args.test_batch_size,
        'kfold': args.kfold, 
        'img_size': args.img_size,
        'epochs': args.num_epochs,
        'arch' : args.arch
    }

    make_submission_file(sub=sample,predictions_proba=predictions_proba, submissions_folder=args.save_resulting_file_to, params=params)