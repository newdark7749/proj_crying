import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from plot_keras_history import show_history, plot_history
import plot_keras_history
from keras.models import Sequential
from keras.layers import Dense,LSTM,Input,Dropout,Flatten,LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
import kerastuner as kt
from sklearn import ensemble
from sklearn import linear_model
from tensorflow import keras
from kerastuner.tuners import Hyperband

def create_model(hp):
    
    input_shape=(40, 401, 1)
    model = Sequential()
    hp_filters=hp.Int('conv_1_filter',min_value=32,max_value=128,step=16)
    hp_kernel_size=hp.Choice('conv_1_kernel',values=[3,5])
    model.add(Conv2D(filters=hp_filters, kernel_size=hp_kernel_size,padding="SAME", activation='relu',kernel_initializer='he_uniform',input_shape=input_shape))
    # model.add(Conv2D(filters=hp_filters, kernel_size=hp_kernel_size,padding="SAME", activation=LeakyReLU(alpha=0.1),kernel_initializer='he_uniform',input_shape=input_shape))
    model.add(MaxPooling2D(padding="SAME"))
    
    hp_filters2=hp.Int('conv_2_filter',min_value=32,max_value=128,step=16)
    hp_kernel_size2=hp.Choice('conv_2_kernel',values=[3,5])
    model.add(Conv2D(filters=hp_filters2,kernel_size=hp_kernel_size2,padding="SAME",activation='relu',kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(padding='SAME'))
    
    # clf
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units=hp.Int('dense_1_units', min_value=32, max_value=64, step=16),activation='relu',kernel_initializer='he_uniform'))
    # output
    model.add(Dense(1,activation='sigmoid'))
    # model.add(Dense(1,activation='tanh'))
    
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

X_train=[]
y_train=[]
X_test=[]
y_test=[]

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
        
        # win_len=0.025 # 25ms
        # frame_stride=0.010 # 10ms
        
        # # wav_length=y.shape[0]/sr
        # input_nfft=int(round(sr*win_len)) # 400
        # input_stride=int(round(sr*frame_stride)) # 160
        
        # n_mels=40 n_fft=400, hop_length=160
        
        S=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
        log_S=librosa.power_to_db(S,ref=np.max)
        
        # print("filename:",filename)
        # print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))
        # print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(log_S)))
        # print("S:",S)
        
        # X_train.append(S)
        X_train.append(log_S)

        # mfcc = librosa.feature.mfcc(y=y,sr=sr)
        # mfcc_avg = []
        # sum_ = 0 
        # for i in mfcc:
        #     for j in i:
        #         sum_ += j
        #     mfcc_avg.append(sum_/10261)
        # X_train.append(mfcc)
        
        if '_noise.wav' in filename:
            y_train.append(noise)
            total_number_of_train_noise+=1
            total_train_noise_data_len+=y.shape[0]/float(sr)
        else:
            y_train.append(cry)
            total_number_of_train_cry+=1
            total_train_cry_data_len+=y.shape[0]/float(sr)
    
    except Exception as e:
        print(filename,e)
        raise
    
# print("==total number of train data files==")
# print("the number of cry files:",total_number_of_train_cry)
# print("the number of noise files:",total_number_of_train_noise)
# print("==total train data length==")
# print("total cry data length:{:.3f}secs".format(total_train_cry_data_len))
# print("total noise data length:{:.3f}secs".format(total_train_noise_data_len))
  

# test 데이터
for filename in os.listdir(TEST_PATH):
    
    try:
        if '.wav' not in filename:
            continue
         
        y,sr = librosa.load(TEST_PATH+filename,sr=16000)
        
        # win_len=0.025 # 25ms
        # frame_stride=0.010
        
        # # wav_length=y.shape[0]/sr
        # input_nfft=int(round(sr*win_len)) # 400
        # input_stride=int(round(sr*frame_stride)) # 160
        
        S=librosa.feature.melspectrogram(y=y,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
        log_S=librosa.power_to_db(S,ref=np.max)
       
        # print("filename:",filename)
        # print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))
        # print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(log_S)))
        # print("S:",S)
       
        X_test.append(S)

        # mfcc = librosa.feature.mfcc(y=y,sr=sr)
        # mfcc_avg = []
        # sum_ = 0 
        # for i in mfcc:
        #     for j in i:
        #         sum_ += j
        #     mfcc_avg.append(sum_/10261)
        # X_test.append(mfcc)
        
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

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
input_shape=X_train.shape[1:]

# X_train shape: (4321, 40, 401, 1)
# X_test shape: (942, 40, 401, 1)
# input_shape: (40, 401, 1)
# print("X_train shape:",X_train.shape)   
# print("X_test shape:",X_test.shape)     
# print("input_shape:",input_shape)       


tuner=Hyperband(    
    create_model,
    objective='val_accuracy',
    max_epochs=100,
    project_name='HB_CNN'
)

tuner.search_space_summary()
tuner.search(X_train,y_train,epochs=100,validation_data=(X_test,y_test))

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f'''
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}
''')

models = tuner.get_best_models(num_models=2)
tuner.results_summary()

# best_model=models[0]
# best_model.build(input_shape=(40,401,1))

best_model = tuner.hypermodel.build(best_hps)
best_model.summary()

his=best_model.fit(X_train, y_train, epochs = 20, validation_data = (X_test, y_test))

yhat = best_model.predict(X_test)
print("yhat:",yhat)


best_model.save("4321CNN_best_model_1.h5")

loss,acc=best_model.evaluate(X_test,y_test)

print("loss:{} acc:{}".format(loss,acc))

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







