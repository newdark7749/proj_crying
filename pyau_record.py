import pyaudio
import wave
import numpy as np
import array

CHUNK = 1600
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 44100
RATE = 16000

RECORD_SECONDS = 0.2
WAVE_OUTPUT_FILENAME = "f1.wav"
WAVE_OUTPUT_FILENAME2 = "f2.wav"
# WAVE_OUTPUT_FILENAME3 = "f3.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Start to record the audio.")

frames = []
frames2=[]
frames3=[]

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    stream.start_stream()
    # data = stream.read(CHUNK)
    data2=abs(np.fromstring(stream.read(CHUNK),dtype=np.int16))
    # data3=array.array('h',data)
    # sDB = sum(data)/len(data)
    # frames.append(data)
    print("data2:",data2)
    print("len data2:",len(data2))
    frames2.append(data2)
    # frames3.append(data3)
    # vol=max(data3)
    # vol=sum(data)/len(data)
    vol2=sum(data2)/len(data2)
    # print("vol:",vol)
    # print("vol2:",vol2)
    
    stream.stop_stream()
# while True:
# print(type(data[0]))
# print(type(data2[0]))
# print(type(data3[0]))
# print(len(data))
# print(len(data2))
# print(len(data3))
# print(data)
# print(data2)
# print(frames[10])
# print(frames2[10])
# print(len(frames))
print(len(frames2))
# print(len(frames3))
print("Recording is finished.")

stream.stop_stream()
stream.close()
p.terminate()

# wf = wave.open('./ex2/'+WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()

# wf2 = wave.open('./ex2/'+WAVE_OUTPUT_FILENAME2, 'wb')
# wf2.setnchannels(CHANNELS)
# wf2.setsampwidth(p.get_sample_size(FORMAT))
# wf2.setframerate(RATE)
# wf2.writeframes(b''.join(frames2))
# wf2.close()

# wf3 = wave.open('./ex2/'+WAVE_OUTPUT_FILENAME3, 'wb')
# wf3.setnchannels(CHANNELS)
# wf3.setsampwidth(p.get_sample_size(FORMAT))
# wf3.setframerate(RATE)
# wf3.writeframes(b''.join(frames3))
# wf3.close()


















