import pandas as pd 
import numpy as np
import os

data_path = '../data/Giz-agri-keywords-data/datasets'
csv_path = '../data/Giz-agri-keywords-data'
sample_df =  pd.read_csv(os.path.join(csv_path, 'SampleSubmission.csv'))
sample_df['fn'] = data_path +'/'+ sample_df['fn']
sample_df.to_csv(os.path.join('.', 'final_test.csv'), index=False)


if __name__ == '__main__':
    print(sample_df['fn'])