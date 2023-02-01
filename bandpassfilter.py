import numpy as np
import scipy
import scipy.io
import scipy.io.wavfile
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter,lfilter

def butter_bandpass(lowcut, highcut,fs, order=5):
    nyq= 0.5*fs
    low= lowcut/nyq
    high= highcut/nyq
    b,a = butter(order, [low,high], btype='band')
    return b,a

def butter_bandpass_filter(data, lowcut,highcut, fs, order=5):
    b,a= butter_bandpass( lowcut, highcut, fs, order)
    y=lfilter(b,a,data)
    return y

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

PATH='./ex/'

# train 데이터
for filename in os.listdir(PATH):
    
    try:
        if '.wav' not in filename:
            continue
         
        y,sr = librosa.load(PATH+filename)
        # sr,y=scipy.io.wavfile.read(PATH+filename)
        
        fs=sr
        order = 10  # order
        cut_off_freq = 200  # cut off frequency
        
        time=y.shape[0]/float(sr)
        t=np.linspace(0, time, len(y), False)  
        freq = np.fft.fftfreq(len(y), 1/1024)
        
        # filtered signal
        # sos = sig.butter(order, [cut_off_freq], 'low', fs=fs, output='sos')  # low pass filter
        # sos = sig.butter(order, [cut_off_freq], 'high', fs=fs, output='sos')  # high pass filter
        sos = sig.butter(order, [300, 600], 'band', fs=fs, output='sos') # band pass filter
        filtered = sig.sosfilt(sos, y)
    
        # raw signal fft
        raw_fft = np.fft.fft(y) / len(y)
        raw_fft_abs = abs(raw_fft)

        # filter signal fft
        filtered_fft = np.fft.fft(filtered) / len(filtered)
        filtered_fft_abs = abs(filtered_fft)

        # plot
        fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2)

        # raw signal plot : 0 row 0 column
        ax00.plot(t, y)
        ax00.set_title('Raw Data Time Domain')
        ax00.set_xlabel('Time [seconds]')
        ax00.set_ylabel('Amplitude')

        # filtered signal plot : 1 row 0 column
        ax10.plot(t, filtered)
        ax10.set_title('Filtered Data Time Domain')
        ax10.set_xlabel('Time [seconds]')
        ax10.set_ylabel('Amplitude')

        # raw signal fft plot : 0 row 1 column
        ax01.stem(freq, raw_fft_abs, use_line_collection=True)
        ax01.set_title('Raw Data Frequency Domain')
        ax01.set_xlabel('Frequency [HZ]')
        ax01.set_ylabel('Amplitude')

        # filtered signal fft plot : 1 row column
        ax11.stem(freq,filtered_fft_abs, use_line_collection=True)
        ax11.set_title('Filtered Data Frequency Domain')
        ax11.set_xlabel('Frequency [HZ]')
        ax11.set_ylabel('Amplitude')

        # plot
        plt.tight_layout()
        plt.show()
        
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




