sudo apt-get install unzip unrar p7zip-full
python3 -m pip install patool -q
python3 -m pip install pyunpack -q
pip intall -q torch
pip install -q pytorch-lightning 
pip install librosa
pip install efficientnet-pytorch
pip install torchaudio

python utils.py --data_path ../data/files --destination_path ../data/files/datasets --extract_files True --kind 7z
python utils.py --data_path  ../data/Giz-agri-keywords-data/datasets --csv_path ../data/Giz-agri-keywords-data --create_train_df True