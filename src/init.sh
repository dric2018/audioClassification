apt-get install unzip unrar p7zip-full
python3 -m pip install patool -q
python3 -m pip install pyunpack -q
pip install -q torch
pip install -q pytorch-lightning 
pip install librosa
pip install efficientnet-pytorch
pip install torchaudio
pip install -U pandas # upgrade pandas
pip install swifter
mkdir ../models
#python utils.py --data_path ../data/files --destination_path ../data/files/datasets --extract_files True --kind 7z
python utils.py --data_path ../data/Giz-agri-keywords-data/datasets --csv_path ../data/Giz-agri-keywords-data --create_train_df True --create_spectrograms True --specs_path ../data/Giz-agri-keywords-data/datasets
python train.py --train_csv_path ../data/Giz-agri-keywords-data/train_test.csv --gpus 1 --test_batch_size 32 --train_batch_size 64 --kfold 3 --num_epochs 25 --img_size 224 --specs_images_path ../data/Giz-agri-keywords-data/datasets/images --save_models_to ../models --seed_value 2020 --lr 0.0023182567385564073
python inference.py --test_csv_path ../data/Giz-agri-keywords-data/final_test.csv --models_path ../models --sample_csv_path ../data/Giz-agri-keywords-data/SampleSubmission.csv --arch 'resnet34' --save_resulting_file_to ../ --test_batch_size 16 --specs_images_path ../data/Giz-agri-keywords-data/datasets/images