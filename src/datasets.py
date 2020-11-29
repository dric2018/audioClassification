import torch 
from torch.utils.data import Dataset, DataLoader
from keras.utils import to_categorical
import librosa
import os
import albumentations as al
from PIL import Image
import cv2
import pandas as pd 
from pytorch_lightning import seed_everything
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, images_path:str,df:pd.DataFrame, transforms=None,  task = 'train', num_classes=193, one_hot=False, **kwargs):
        super(AudioDataset, self).__init__()

        self.task = task 
        self.df = df
        try:
            self.class_dict = {label:idx for idx,label in enumerate(self.df.label.unique().tolist())}
        except:
            pass
        self.transforms = transforms
        self.one_hot = one_hot
        self.images_path = images_path
        self.num_classes = num_classes
        self.log_specs = os.listdir(self.images_path)


    def __getitem__(self, index):
        wav_path = self.df.iloc[index].fn
        file_ = wav_path.split('/')[-1].split('.wav')[0]
        file_path = self.images_path +'/'+ file_ +'.png'
        
        # load spectrogram
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        sample = {'image' : torch.tensor(img, dtype=torch.float)}

        if self.task == 'train':
            label = self.df.iloc[index].label
            if self.one_hot:
                sample.update({
                    'label' : torch.tensor(to_categorical(label, self.num_classes), dtype=torch.float)
                })            
            else:
                sample.update({
                    'label' : torch.tensor(label, dtype=torch.long)
                })
        return sample


    def __len__(self):
        return len(self.df)



if __name__ == '__main__':

    seed_val = 2020
    _ = seed_everything(seed_val)
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
    #train_df = pd.read_csv('../data/Giz-agri-keywords-data/final_train.csv')
    #ds = AudioDataset(images_path='../data/Giz-agri-keywords-data/datasets/images', df=train_df, transforms=data_transforms['train'])
    #dl = DataLoader(dataset=ds, shuffle=True, batch_size=32, num_workers=os.cpu_count())

    test_df = pd.read_csv('../data/Giz-agri-keywords-data/final_test.csv')
    test_ds = AudioDataset(images_path='../data/Giz-agri-keywords-data/datasets/images', task='test', df=test_df, transforms=data_transforms['test'])
    test_dl = DataLoader(dataset=test_ds, shuffle=False, batch_size=16)
    for data in test_dl:
        print(data['image'].shape)
        break
    