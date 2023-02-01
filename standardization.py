import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import scipy
import scipy.io
import scipy.io.wavfile
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os
import random
import pickle
import time
import pandas as pd

# start=time.time()

X_train=[]
y_train=[]
X_test=[]
y_test=[]

# test_name=[]
# test_file_cnt=0

# # total 0,1 count
# total_number_of_train_cry=0
# total_number_of_train_noise=0

# # total audio file length
# total_train_cry_data_len=0
# total_train_noise_data_len=0

cry=0
noise=1
train_name=[]
test_name=[]
nbits=16

TRAIN_PATH='./cle/train_/'
TEST_PATH='./cle/test_/'

# train 데이터
for filename in os.listdir(TRAIN_PATH):
    
    try:
        if '.wav' not in filename:
            continue
         
        l_wave,sr = librosa.load(TRAIN_PATH+filename,sr=None)
    
        # sr,s_wave=scipy.io.wavfile.read(TRAIN_PATH+filename)
        
        # tmp_s_wave=l_wave * ( 2**(nbits-1))
        
        # print("cal==")
        # print(tmp_s_wave)
        # print("ori==")
        # print(s_wave)
        X_train.append(l_wave)
        
        # sf.write(save_file,ny,sr)
        train_name.append(filename)
        
        if '_noise.wav' in filename:
            y_train.append(noise)
            # total_number_of_train_noise+=1
            # total_train_noise_data_len+=y.shape[0]/float(sr)
        else:
            y_train.append(cry)
            # total_number_of_train_cry+=1
            # total_train_cry_data_len+=y.shape[0]/float(sr)
    
    except Exception as e:
        print(filename,e)
        raise

# test 데이터
for filename in os.listdir(TEST_PATH):
    
    try:
        if '.wav' not in filename:
            continue
         
        y,sr = librosa.load(TEST_PATH+filename,sr=None)
        
        X_test.append(y)
        
        # sf.write(save_file,ny,sr)
        
        test_name.append(filename)
        
        
        if '_noise.wav' in filename:
            # test_file_cnt+=1
            # test_name.append(filename[:11])
            y_test.append(noise)
        else:
            # test_file_cnt+=1
            # test_name.append(filename[:11])
            y_test.append(cry)
        
    except Exception as e:
        print(filename,e)
        raise

# print("==total number of train data files==")
# print("the number of cry files:",total_number_of_train_cry)
# print("the number of noise files:",total_number_of_train_noise)
# print("==total train data length==")
# print("total cry data length:",total_train_cry_data_len,"secs")
# print("total noise data length:",total_train_noise_data_len,"secs")

# df1=pd.DataFrame(X_train)
# print(df1.describe())
# plt.boxplot(df1)
# plt.show()

scal=StandardScaler()

scal.fit(X_train)

pickle.dump(scal,open('./std_scaler_1_13.pkl','wb'))


X_train_s=scal.transform(X_train)
X_test_s=scal.transform(X_test)

X_train=np.array(X_train)
X_test=np.array(X_test)

# print("==X_train==")
# print("X_train shape:",X_train.shape)
# print(X_train)


# print("==X_train_s==")
# print("_s shape:",X_train_s.shape)
# print(X_train_s)

# print("X_train_s[0]leng:",len(X_train_s[0]))

sr=16000
TRAIN_PATH='./cle/std_train_/'
TEST_PATH='./cle/std_test_/'

for fn in range(len(X_train_s)):
    sf.write(TRAIN_PATH+train_name[fn],X_train_s[fn],sr)
for fin in range(len(X_test_s)): 
    sf.write(TEST_PATH+test_name[fin],X_test_s[fin],sr)