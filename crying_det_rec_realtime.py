import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from keras.models import Sequential
from keras.layers import Dense,LSTM,Input,Dropout,Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import os
import random
import pickle
import time
import pandas
import tensorflow as tf
from tensorflow import keras
# import kerastuner as kt
import joblib
import os
import random
import pyaudio


def more_than_half(nums):
    first = []
    second = []
    for i in nums:
        if i == 0:
            first.append(i)
        else:
            second.append(i)
    if len(first) > len(second):
        return 0
    else:
        return 1
# 2

import tensorflow as tf
def Crying_Detection(filename): 
    cry=0
    noise=1

    win_len=0.025 # 25ms
    frame_stride=0.010
    sr=16000
    
    input_nfft=int(round(sr*win_len)) # 400
    input_stride=int(round(sr*frame_stride)) # 160

    new_model = tf.keras.models.load_model('./cnn.h5')
    
    sum_snr = 0
    sum___ = []
    
    X_test = []
    
    y,sr = librosa.load(filename,sr=16000)

    S=librosa.feature.melspectrogram(y=y,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
    log_S=librosa.power_to_db(S,ref=np.max)

    X_test.append(S)
    X_test = np.array(X_test)
    X_test11 = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    new_hat = new_model.predict(X_test11)

    #print(new_hat[0][0])
    if new_hat[0][0] < 0.001: # 100
        return 0

    else:
        return 1

import wave
def save_Sound(sframe):
    
    frames = sframe #[]

    WAVE_OUTPUT_FILENAME = "./sample.wav"  #"./sample.wav"
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()  

    return Crying_Detection(WAVE_OUTPUT_FILENAME)

import time

frames = []

FORMAT = pyaudio.paInt16
    
CHANNELS = 1

RATE = 16000

CHUNK = 1600

RECORD_SECONDS = 5

WAVE_OUTPUT_FILENAME = "./sample.wav"

# DEVICE = 5
DEVICE=1

DBTHRESHOLD = 1000

audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16, 

                channels=CHANNELS, 

                rate=RATE,

                input=True, 

                input_device_index=DEVICE,

                frames_per_buffer=CHUNK)


r_list = []
avg_list = []
fsum=0
while True:
    
    stream.start_stream()
    data = abs(np.fromstring(stream.read(CHUNK),dtype=np.int16))
    sDB = sum(data)/len(data)
    # print(">>>>sDB",sDB)
    frames.append(data)
    stream.stop_stream()
    if len(frames) % 10 == 0 and len(frames) >= 50:
        frames = frames[10:]
        r_list.append(save_Sound(frames))
        
        if len(r_list) == 6:
            r_list = r_list[1:]
            print("r_list : ", r_list)
            avg_list.append(more_than_half(r_list))
            if len(avg_list) == 6:
                avg_list = avg_list[1:]
                print(avg_list)
                if more_than_half(avg_list) == 0:
                    if sDB >= DBTHRESHOLD:
                        print("CRYING - take care a Baby")
                        continue
                else:
                    if sDB >= DBTHRESHOLD:
                        print("WAIT")
                        continue
            else:
                if more_than_half(r_list) == 0:
                    if sDB >= DBTHRESHOLD:
                        print("CRYING - take care a Baby")
                        
                else:
                    if sDB >= DBTHRESHOLD:
                        print("WAIT")