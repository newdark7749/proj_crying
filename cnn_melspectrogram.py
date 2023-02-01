import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
# import soundfile as sf
# from plot_keras_history import show_history, plot_history
# import plot_keras_history
from keras.models import Sequential
from keras.layers import Dense,LSTM,Input,Dropout,Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from sklearn import svm
# from sklearn import ensemble
# from sklearn import linear_model
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
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
# import keras_tuner as kt
# from keras_tuner.tuners import Hyperband


def create_model():
    
    input_shape=(40, 401, 1)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3,padding="SAME", activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(padding="SAME"))
    
    model.add(Conv2D(64,kernel_size=3,padding="SAME",activation='relu'))
    model.add(MaxPooling2D(padding='SAME'))
    
    # clf
    model.add(Flatten())
    # model.add(Dropout(0.3))
    # model.add(Dropout(0.6))
    
    model.add(Dense(128,activation='relu'))
    # output
    model.add(Dense(1,activation='sigmoid'))
    
    return model
    
    

# n_fft=2048
# win_length=2048
# hop_length=1024
# n_mels=128
# D=np.abs(librosa.stft(y,n_fft=n_fft,win_length=win_length,hop_length=hop_length))
# mel_spec=librosa.feature.melspectrogram(S=D,sr=sr,n_mels=n_mels,hop_length=hop_length,win_length=win_length)
# librosa.display.specshow(librosa.amplitude_to_db(mel_spec,ref=0.00002),sr=sr,hop_length=hop_length,y_axis='mel',x_axis='time',cmap=cm.jet)
# plt.colorbar(format='%2.0fdB')
# plt.show()

X_train=[]
y_train=[]
X_test=[]
y_test=[]

X_train1=[]
X_test1=[]

test_name=[]
test_file_cnt=0

# total 0,1 count
total_number_of_train_cry=0
total_number_of_train_noise=0

# total audio file length
total_train_cry_data_len=0
total_train_noise_data_len=0

cry=0
noise=1

win_len=0.025 # 25ms
frame_stride=0.010
sr=16000

# wav_length=y.shape[0]/sr
input_nfft=int(round(sr*win_len)) # 400
input_stride=int(round(sr*frame_stride)) # 160

TRAIN_PATH='./cle/std_train_/'
TEST_PATH='./cle/std_test_/'

# train 데이터
for filename in os.listdir(TRAIN_PATH):
    
    try:
        if '.wav' not in filename:
            continue
         
        y,sr = librosa.load(TRAIN_PATH+filename,sr=16000)
        
        
        S=librosa.feature.melspectrogram(y=y,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
        S1=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
        
        print(S.shape)
        print(S1.shape)
        print(S)
        print(S1)
        
        breakpoint()
        
        log_S=librosa.power_to_db(S,ref=np.max)
        
        # mfcc=librosa.feature.mfcc(y=y,sr=sr,n_fft=input_nfft,hop_length=input_stride)
        # mfcc_l=librosa.feature.mfcc(S=log_S,sr=sr)
             
        
        breakpoint() 
        
        X_train.append(S)
        X_train1.append(log_S)

        
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
         
        y,sr = librosa.load(TEST_PATH+filename,sr=16000)
                
        S=librosa.feature.melspectrogram(y=y,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
        log_S=librosa.power_to_db(S,ref=np.max)

        
        X_test.append(S)
        X_test1.append(log_S)
        

        if '_noise.wav' in filename:
            y_test.append(noise)
            # total_number_of_test_noise+=1
            # total_test_noise_data_len+=y.shape[0]/float(sr)
        else:
            y_test.append(cry)
            # total_number_of_test_cry+=1
            # total_test_cry_data_len+=y.shape[0]/float(sr)
    
    except Exception as e:
        print(filename,e)
        raise

X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

X_train1=np.array(X_train1)
X_test1=np.array(X_test1)

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

X_train1=X_train1.reshape(X_train1.shape[0],X_train1.shape[1],X_train1.shape[2],1)
X_test1=X_test1.reshape(X_test1.shape[0],X_test1.shape[1],X_test1.shape[2],1)

input_shape=X_train.shape[1:]


# X_train shape: (4321, 40, 401, 1)
# X_test shape: (1138, 40, 401, 1)
# input_shape: (40, 401, 1)

# breakpoint()

model=create_model()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

model1=create_model()
model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model1.summary()

# his=model.fit(X_train,y_train,epochs=20,verbose=2,batch_size=10)
his=model.fit(X_train,y_train,epochs=30,verbose=2,batch_size=10,validation_data=(X_test,y_test))
his1=model1.fit(X_train1,y_train,epochs=30,verbose=2,batch_size=10,validation_data=(X_test1,y_test))


model.save("1_15_cnn_nodropout.h5")
model1.save("1_15_log_cnn_nodropout.h5")

loss,acc=model.evaluate(X_test,y_test)
loss1,acc1=model1.evaluate(X_test1,y_test)

print("loss:{} acc:{}".format(loss,acc))
print("loss1:{} acc1:{}".format(loss1,acc1))

def vis(history,name) :
    plt.title(f"{name.upper()}")
    plt.xlabel('epochs')
    plt.ylabel(f"{name.lower()}")
    value = history.history.get(name)
    val_value = history.history.get(f"val_{name}",None)
    epochs = range(1, len(value)+1)
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    if val_value is not None :
        plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2) , fontsize=10 , ncol=1)
    
def plot_history(history) :
    key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
    plt.figure(figsize=(12, 4))
    for idx , key in enumerate(key_value) :
        plt.subplot(1, len(key_value), idx+1)
        vis(history, key)
    plt.tight_layout()
    plt.show()

plot_history(his)
plot_history(his1)

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.title('loss of cnn ', fontsize= 15)
# plt.plot(his.history['loss'], 'b-', label='loss')
# plt.plot(his.history['val_loss'],'r--', label='val_loss')
# plt.xlabel('Epoch')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.title('accuracy of cnn ', fontsize= 15)
# plt.plot(his.history['accuracy'], 'g-', label='accuracy')
# plt.plot(his.history['val_accuracy'],'k--', label='val_accuracy')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show







