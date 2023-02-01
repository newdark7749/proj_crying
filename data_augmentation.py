import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
from sklearn.svm import SVR
from sklearn.svm import SVC
import os
import random
import pickle


def plot_time_series(data,fname):
    fig = plt.figure(figsize=(10,4))
    plt.title(str(fname))
    plt.ylabel("Amplitude")
    plt.plot(np.linspace(0,data.shape[0]/float(22050),len(data)),data)
    plt.show()
    
def stretch_sound(data,sr,rate,filename):
    stretch_data=librosa.effects.time_stretch(y=data,rate=rate)
    stretch_data_name="stretch_"+str(rate)+'_'+filename
    plot_time_series(stretch_data,stretch_data_name)
    sf.write(PATH+stretch_data_name,stretch_data,sr)
    # return data

def adding_white_noise(data,sr,noise_rate,filename):
    wn=np.random.randn(len(data))
    data_wn=data+noise_rate*wn
    white_noise_data_name="addwn_"+str(noise_rate)+'_'+filename
    plot_time_series(data_wn,white_noise_data_name)
    sf.write(PATH+white_noise_data_name,data_wn,sr)
    # return data
    
def white_noise(noise_data_time_sec,sr,noise_rate):
    
    wn=np.random.randn(noise_data_time_sec*sr)
    data_wn=noise_rate*wn
    white_noise_data_name="noise_"+str(noise_rate)+'.wav'
    # plot_time_series(data_wn,white_noise_data_name)
    sf.write(PATH+white_noise_data_name,data_wn,sr)
    # return data_wn

def reverse_sound(data,sr,filename):
    data_len = len(data)
    r_data = np.array([data[len(data)-1-i] for i in range(len(data))])
    r_data_name="reversedata_"+filename
    plot_time_series(r_data,r_data_name)   
    sf.write(PATH+r_data_name,r_data,sr)
    # return data

def polarity_inv_sound(data,sr,filename):
    
    temp_numpy = (-1)*data
    inv_data_name="invdata_"+filename
    plot_time_series(temp_numpy,inv_data_name)
    sf.write(PATH+inv_data_name,temp_numpy,sr)
    
    # return data

def shifting_sound(data,sr,roll_rate,filename):
    data_roll=np.roll(data,int(len(data)*roll_rate))
    roll_data_name="roll_"+str(roll_rate)+"_"+filename
    plot_time_series(data_roll,roll_data_name)
    sf.write(PATH+roll_data_name,data_roll,sr)

# f_name='awake_2.wav'
# data,sr=librosa.load("./ex/"+f_name,sr=22050)
# plot_time_series(data,f_name)
# reverse_sound(data,sr,f_name)
# polarity_inv_sound(data,sr,f_name)

# print(librosa.__version__)

PATH='./ex1/'

# augmentation
for filename in os.listdir(PATH):
    try:
        if '.wav' not in filename:
            continue
        y,sr = librosa.load(PATH+filename,sr=None)
        # plot_time_series(y,filename)
        # stretch_sound(y,sr,0.65,filename)
        # stretch_sound(y,sr,1.3,filename)
        # adding_white_noise(y,sr,0.003,filename)
        # adding_white_noise(y,sr,0.007,filename)
        # polarity_inv_sound(y,sr,filename)
        shifting_sound(y,sr,0.1,filename)
        shifting_sound(y,sr,0.3,filename)
        shifting_sound(y,sr,0.6,filename)
        shifting_sound(y,sr,0.75,filename) 
        shifting_sound(y,sr,0.88,filename)
        
        # white_noise(10,16000,0.05)

    except Exception as e:
        print(filename,e)
        raise





