import torch 
from torch.data.utils import Dataset, DataLoader
from keras.utils import to_categorical
import librosa


class AudioDataset(Dataset):
    def __init__(self, files_path:str,  task = 'train', use_features='mel_spec', num_classes=2, one_hot=True, **kwargs, **args):
        super(AudioDataset, self).__init__()

        self.task = task 
        self.files_path = files_path
        self.num_classes = num_classes
        self.melSpecs = []
        self.mfccs = []
        self.logFBanks = []
        self.targets = []

        # extract features from audio


    def __getitem__(self, index):
        pass


    def __len__(self):
        pass

    