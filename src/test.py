import pandas as pd 
import numpy as np
import os
from tqdm import tqdm 
import librosa
import matplotlib.pyplot as plt 
import swifter
from scipy import signal



def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):

    """
    Borrowing log spec function from https://www.kaggle.com/davids1992/data-visualization-and-investigation
    """
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def wav2img(wav_path, targetdir='', figsize=(4,4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """
    fig = plt.figure(figsize=figsize)    
    # use soundfile library to read in the wave files
    sound, samplerate  = librosa.load(wav_path)
    _, spectrogram = log_specgram(sound, samplerate)
    
    ## create output path
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir +'/'+ output_file
    #plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.imsave('%s.png' % output_file, spectrogram)
    plt.close()

    return output_file+'.png'




data_path = '../data/Giz-agri-keywords-data/datasets'

img_dir = data_path+'/images'



if __name__ == '__main__':
    # create test_spectrograms
    sample_csv_path = '../data/Giz-agri-keywords-data/SampleSubmission.csv'
    sample = pd.read_csv(sample_csv_path)
    sample['fn'] = data_path +'/'+ sample['fn']
    sample['spec_path'] = img_dir

    for row in tqdm(sample.iterrows(), total=len(sample), desc='Creating specs'):
        output_file = wav2img(wav_path=row[1].fn, targetdir=img_dir)
        sample.at[row[0], 'spec_path'] = output_file

    sample.to_csv(os.path.join('../data/Giz-agri-keywords-data/', 'final_test.csv'), index=False)
    print(sample.head(5))