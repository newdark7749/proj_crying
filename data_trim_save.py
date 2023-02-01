import joblib
import librosa
import numpy as np
import soundfile as sf
import os
import random
import pickle
import time
import math,sys
import pandas
import scipy
import scipy.io
import scipy.io.wavfile


def data_save_SEC(audio_file,save_file,SEC):
    
    y,sr=librosa.load(audio_file,sr=16000)
    # sr,y=scipy.io.wavfile.read(audio_file)

    trim=SEC*sr
    trim=int(trim)
    
    if y.shape[0]>trim:
        ny=y[:trim]
    elif y.shape[0]<trim:
        ny=np.concatenate((y,y))
        ny_len=ny.shape[0]
        while ny_len<trim:
            ny=np.concatenate((ny,y))
            ny_len=ny.shape[0]
        ny=ny[:trim]
    else:
        ny=y

    sf.write(save_file,ny,sr)
    
cry=0
noise=1

PATH='./temp/'
SAVE_PATH='./temp/cle/'

SEC=4

for filename in os.listdir(PATH):
    
    try:
        if '.wav' not in filename:
            continue
        
        audio_file=PATH+filename
        
        data_save_SEC(audio_file,SAVE_PATH+filename,SEC)

    except Exception as e:
        print(filename,e)
        raise


