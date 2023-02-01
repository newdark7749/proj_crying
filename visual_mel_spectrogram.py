import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


win_len=0.025 # 25ms
frame_stride=0.010 # 10ms
sr=16000

# wav_length=y.shape[0]/sr
input_nfft=int(round(sr*win_len)) # 400
input_stride=int(round(sr*frame_stride)) # 160


# Load the audio file
filename = './ex1/10_s.wav'
y, sr = librosa.load(filename,sr=16000)

# Compute the Mel Spectrogram
# S=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,n_fft=input_nfft,hop_length=input_stride,fmax=8000)
S=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
# log_S=librosa.power_to_db(S,ref=np.max)

# Plot the Mel Spectrogram
plt.figure(figsize=(10, 4))
plt.title('Mel Spectrogram')
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# import os
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
# import pandas as pd

# def white_noise(noise_data_time_sec=10,sr=16000,noise_rate=0.05):
    
#     wn=np.random.randn(noise_data_time_sec*sr)
#     data_wn=noise_rate*wn
#     S=librosa.feature.melspectrogram(y=data_wn,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
#     log_S=librosa.power_to_db(S,ref=np.max)
#     # white_noise_data_name="noise_"+str(noise_rate)+'.wav'
#     # sf.write(TRAIN_PATH+white_noise_data_name,data_wn,sr)
#     return log_S

# def mels_time_plt(log_S,sr,filename):
#     plt.figure(figsize=(10,4))
#     librosa.display.specshow(log_S,sr=sr,hop_length=input_stride,x_axis='time',y_axis='mel')
#     plt.xlabel("Time")
#     plt.ylabel("Mel")
#     plt.title(filename+"_Mel-Spectrogram")
#     # plt.tight_layout()
#     # plt.savefig("Mel-Spectrogram ex.png")
#     plt.colorbar(format='%+2.0f dB')
#     plt.show()

# def plot_time_series(data,fname):
#     fig = plt.figure(figsize=(10,4))
#     plt.title(str(fname))
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.plot(np.linspace(0,data.shape[0]/float(16000),len(data)),data)
#     plt.show()
    
# sr=16000

# win_len=0.025 # 25ms
# frame_stride=0.010 # 10ms

# # wav_length=y.shape[0]/sr
# input_nfft=int(round(sr*win_len)) # 400
# input_stride=int(round(sr*frame_stride)) # 160

# X_train=[]
# tmp=[]
# tmp2=[]

# PATH='./ex1/'

# for filename in os.listdir(PATH):
    
#     try:
#         if '.wav' not in filename:
#             continue
         
#         y,sr = librosa.load(PATH+filename,sr=16000)
        
#         S=librosa.feature.melspectrogram(y=y,n_mels=40,n_fft=input_nfft,hop_length=input_stride)
#         # log_S=librosa.power_to_db(S,ref=np.max)
#         # librosa.feature.inverse.mel_to_audio()
        
#         # for _ in range(5):
#         #     log_S=white_noise(10,sr,0.03)
#         #     X_train.append(log_S)
#         # X_train.append(S)
        
#         # print("S:",S)
#         # print(S.shape)
#         # print("filename:",filename)
#         # print("log_S:",log_S)
#         # print(log_S.shape)
#         # for x in log_S:
#         #     tmp.append(min(x))
#         #     tmp2.append(max(x))
            
#         # for x in S:
#         #     tmp.append(min(x))
#         #     tmp2.append(max(x))
            
#         # X_train=np.array(X_train)
#         # print(X_train.shape)
#         # print(X_train)

#         # mels_time_plt(log_S,sr,filename)
    

#         # plot_time_series(y,filename)
#         # mels_time_plt(white_noise())

#         # func
#         # plt.figure(figsize=(10,4))
#         # librosa.display.specshow(log_S,sr=sr,hop_length=input_stride,x_axis='time',y_axis='mel')
#         # plt.xlabel("Time")
#         # plt.ylabel("Mel")
#         # plt.title("Mel-Spectrogram")
#         # # plt.tight_layout()
#         # # plt.savefig("Mel-Spectrogram ex.png")
#         # plt.colorbar(format='%+2.0f dB')
#         # plt.show()
        
#         # print("filename:",filename)
#         # print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))
#         # print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(log_S)))
#         # print("S:",S)
        

#     except Exception as e:
#         print(filename,e)
#         raise

# # print("min")
# # print(tmp)
# # print("max")
# # print(tmp2)

# X_train=np.array(X_train)

# print(X_train.shape)




