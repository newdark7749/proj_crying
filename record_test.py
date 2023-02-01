import pyaudio
import wave
import numpy as np
import os
import librosa
import joblib
import tkinter
import tkinter.font
import time
import tensorflow as tf
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

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

def callBandPassSample(a):
    audio_wave_data, sr = librosa.load(a) 
    audio = audio_wave_data * (2 ** 15 - 1) / np.max(np.abs(audio_wave_data))
    YY = []
    for i in range(1, 4001, 200):
        y0= butter_bandpass_filter(audio, i,i+200, sr, 3)
        YY.append(y0)
    
    return YY

FORMAT = pyaudio.paInt16

CHANNELS = 1

RATE = 16000

CHUNK = 1024
 
RECORD_SECONDS = 5

limit_db = 1000

WAVE_OUTPUT_FILENAME = "./11.wav"

DEVICE = 1

cry_sw=0 
Sound_1 = 0;    # 사운드 모듈 1
Sound_2 = 0;    # 사운드 모듈 2
count = 0;       # 사운드 모듈 데이터 갯수
result = 0;     # 데이터의 연산 값
stack = 3;       # 울음소리 감지
S_stack = 0;     # stack 이전값 저장
noise = 0;      # stack 초기화
sum_snr = 0
audio = pyaudio.PyAudio()

# start Recording

stream = audio.open(format=pyaudio.paInt16, 

                channels=CHANNELS, 

                rate=RATE, 

                input=True, 

                input_device_index=DEVICE,

                frames_per_buffer=CHUNK)

loud = []
sum__ = 0
sum___ = []
fre_list = []

try:
    while True:

        data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
        #print(int(np.average(np.abs(data))))

        if(count >= 10):
            count = 0
            result = Sound_1
            result = result // 10
            if(result >= limit_db):
                stack = stack +1
            print("Sound Detecting.. : (", result, ")")

            if(stack == S_stack):
                noise = noise + 1
            if(stack > S_stack):
                noise = 0      # stack에 새로운 값 입력시 noise 초기화
            
            S_stack = stack     #S_stack에 이전 stack값 저장
            if(noise >=10):
                noise=0
                stack=0

            if(stack > 3): #stack에 따라 울음감지
                print("**Take care of Baby**")
                stack = stack - 2
                cry_sw = 1

    
            Sound_2 = Sound_1
            Sound_1 = 0
            count = 0


        count = count +1
        Sound_add = int(np.average(np.abs(data)))

        Sound_1 = abs(Sound_add) + Sound_1
        
       
        
        if cry_sw == 1:
            cry_sw = 0
            #print ("recording...")
            frames = []
            
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

                data = stream.read(CHUNK)

                frames.append(data)

            #print ("finished recording")


            sum__ = 0

            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()  

            X_test = []
            y , sr = librosa.load(WAVE_OUTPUT_FILENAME) # file_name librosa.load() : 오디오 파일을 로드한다. 'crysound.wav'
            #y = callBandPassSample(WAVE_OUTPUT_FILENAME)
            y_o = y
            y = callBandPassSample(WAVE_OUTPUT_FILENAME)
            X_test.append(y)

            X_test11 = X_test
            for i in range(len(X_test11)):
                for j in range(len(X_test11[i])):
                    X_test11[i][j] = X_test11[i][j][:40000]
                    
            X_test11 = np.array(X_test11)
            X_test11 = X_test11.reshape((X_test11.shape[0], X_test11.shape[1], X_test11.shape[2])) # LSTM
            
            for i in range(len(X_test11)):
                for j in range(len(X_test11[i])):
                    X_test11[i][j] = X_test11[i][j][:40000]

            #X_test22 = X_test1.reshape((X_test1.shape[0], X_test1.shape[1], 1, 1)) # CNN, CRNN 
            #model.save('LSTM_Crying.h5')
            
            new_model = tf.keras.models.load_model('./BPFLSTM_Crying.h5')  # BPFLSTM_Crying
            #clf_from_joblib2 = joblib.load('./LSTMmodel.pkl')
            #new_model1 = tf.keras.models.load_model('학부생 Crying detection\CNN_Crying.h5')
            #new_model2 = tf.keras.models.load_model('학부생 Crying detection\RCNN_Crying.h5')
            
            #for i in range(len(X_test11)):

            #X_test111 = X_test11[0][:10000]
            
             # SNR 추가
            WIN = int(len(y_o) / ((len(y_o)/sr)/0.5))

            for i in range(len(y_o)//WIN):

                fft = np.fft.fft(y_o[WIN*i:WIN*(i+4)])

                # 복소공간 값 절댓갑 취해서, magnitude 구하기
                magnitude = np.abs(fft) 

                # Frequency 값 만들기
                f = np.linspace(0,sr,len(magnitude))

                # 푸리에 변환을 통과한 specturm은 대칭구조로 나와서 high frequency 부분 절반을 날려고 앞쪽 절반만 사용한다.
                left_spectrum = magnitude[:int(len(magnitude)/2)]
                left_f = f[:int(len(magnitude)/2)]
                '''
                plt.figure(figsize=(16,6))
                plt.plot(left_f, left_spectrum)
                plt.xlabel("Frequency")
                plt.ylabel("Magnitude")
                plt.title("Power spectrum")
                '''
                sum_2 = 0
                sum_ = 0
                for i in range(len(left_f)):
                    sum_2 += left_spectrum[i]
                    if left_f[i] > 1000 and left_f[i] < 4000:
                        sum_ += left_spectrum[i]

                #print("SNR : ", sum_ / sum_2)
                sum___.append(sum_ / sum_2)
            for i in sum___:
                sum_snr += i
            #print("avg_snr : ", sum_snr / len(sum___))
            
            if sum_snr / len(sum___) < 1:
                continue
            #print(sum___ // )
                #sum__.append(sum_/sum_2)

            ### SNR Part ###
        
        
            #X_test111 = X_test111.reshape((1, 10000, 1))    
            new_hat = new_model.predict(np.array(X_test11))
            #KNN
            
            print("new_hat[0][0]:",new_hat[0][0])
            if new_hat[0][0] <= -400: # 100
                    print("LSTM : Crying")
            
            else:
                    # continue
                    print("LSTM : Noise")
            
            #break

            #if os.path.isfile(WAVE_OUTPUT_FILENAME):
                #os.remove(WAVE_OUTPUT_FILENAME)
            

            #print( mfcc.shape )


            # stop Recording
        
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    audio.terminate()



# import pyaudio
# import wave
# import numpy as np

# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# # RATE = 44100
# RATE = 16000

# RECORD_SECONDS = 1
# WAVE_OUTPUT_FILENAME = "rec_ex.wav"

# p = pyaudio.PyAudio()

# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

# loud = []
# sum__ = 0
# sum___ = []
# fre_list = []

# print("Start to record the audio.")

# frames = []

# cnt=0
# # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

# while True:
#     data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
#     print("cnt:",cnt)
#     # print("i:",i)
#     print("data:",data)
#     sound_1=int(np.average(np.abs(data)))
#     print("sount_1:",sound_1)
#     frames.append(data)

# print("Recording is finished.")

# stream.stop_stream()
# stream.close()
# p.terminate()

# wf = wave.open('./ex/'+WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()