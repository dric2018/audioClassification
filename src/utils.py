import os
import shutil
from zipfile import ZipFile


def extract_files(data_path:str, destination_path:str):
    files = os.listdir(data_path)
    dest = os.path.join(destination_path)

    os.makedirs(dest, exist_ok=True)

    