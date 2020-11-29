import os
import shutil
from zipfile import ZipFile
from tqdm import tqdm 
from pyunpack import Archive
import argparse
import pandas as pd 
import numpy as np
import librosa


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',  type=str, help='data source directory')
parser.add_argument('--destination_path',  type=str, help='data destination directory')
parser.add_argument('--extract_files', default=False, type=bool, help='execute extraction or not')
parser.add_argument('--kind', default='7z', type=str, help='For .7z files extraction')
parser.add_argument('--create_train_df', default=True, type=bool, help='Create a training dataframe or not')
parser.add_argument('--csv_path', type=str, help='Csv files path')


def extract_files(data_path:str, destination_path:str):
    files = os.listdir(data_path)
    dest = os.path.join(destination_path)

    os.makedirs(dest, exist_ok=True)

    for fn in tqdm(files):
        if fn.split('.')[-1] == "zip":
            try :
                with ZipFile(os.path.join(data_path, fn), "r") as zip_ref:
                    for file_ in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc="Extrating files"):

                        # Extract each file to another directory
                        # If you want to extract to current working directory, don't specify path
                        zip_ref.extract(member=file_, path=dest)
                    print(f'[INFO] successfully extracted files from {fn}')

            except Exception as ex:
                print(f'[ERROR] {ex}')


def extract_files_v1(data_path:str, destination_path:str):
    files = os.listdir(data_path)
    dest = os.path.join(destination_path)

    os.makedirs(dest, exist_ok=True)

    for fn in tqdm(files):
        if fn.split('.')[-1] == "7z":
            try :
                print(f'[INFO] Extracting files from {fn}')

                tqdm(Archive(os.path.join(data_path, fn)).extractall(dest), desc=len(os.listdir(os.path.join(data_path, fn))))
                print(f'[INFO] successfully extracted files from {fn}')

            except Exception as ex:
                print(f'[ERROR] {ex}')



def calc_duration(file_path):
    signal, sr = librosa.load(file_path)
    return signal.shape[0] / sr
    


def create_train_dataframe(csv_path, data_path):
    if ('Train.csv' in os.listdir(csv_path))  or ('train.csv' in os.listdir(csv_path)):
        try:
            df = pd.read_csv(os.path.join(csv_path, 'Train.csv'))
        except:
            df = pd.read_csv(os.path.join(csv_path, 'train.csv'))


        folder_list = os.listdir(data_path)
        files_list = []
        labels = []

        for folder in folder_list:
            if folder != 'audio_files':
                try:
                    keywords = os.listdir(os.path.join(data_path, folder))
                    for keyword in keywords:
                        files = os.listdir(os.path.join(data_path, folder, keyword))
                        files_list += [os.path.join(folder, keyword, fn) for fn in files ]
                        labels += [keyword for _ in range(len(os.listdir(os.path.join(data_path, folder, keyword)))) ]
                except:
                    pass
                
        

        df = df.append(pd.DataFrame({
            'fn' : files_list,
            'label' : labels
        }), ignore_index=True)


        df['fn'] = data_path +'/'+ df['fn']
        df['duration'] = df.apply(lambda row : calc_duration(row.fn))
        df.to_csv(os.path.join(csv_path, 'final_train.csv'), index=False)

        try:
            samplde_df =  pd.read_csv(os.path.join(csv_path, 'SampleSubmission.csv'))
            sample_df['fn'] = data_path +'/'+ sample_df['fn']
        except:
            pass



def 

if __name__ == '__main__':

    args = parser.parse_args()

    if args.extract_files:
        if args.kind == 'zip':
            try:
                extract_files(args.data_path, args.destination_path)
            except Exception as ex:
                raise ex
        else:
            try:
                extract_files_v1(args.data_path, args.destination_path)
            except Exception as ex:
                raise ex

    if args.create_train_df:
        try:
            create_train_dataframe(args.csv_path, args.data_path)
        except Exception as ex:
            print(ex)



