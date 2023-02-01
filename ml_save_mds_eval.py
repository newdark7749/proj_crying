import joblib
from matplotlib.cbook import flatten
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import os
import random
import pickle
import time
import pandas as pd

def get_clf_eval(y_test,pred):
    accuracy=accuracy_score(y_test,pred)
    precision=precision_score(y_test,pred)
    recall=recall_score(y_test,pred)
    f1_sc=f1_score(y_test,pred)
    
    print("accuracy:{:.3f},Precision:{:.3f}, Recall:{:.3f}, f1_score:{:.3f}".format(accuracy,precision,recall,f1_sc))

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

TRAIN_PATH='./temp/cle/'
TEST_PATH='./temp/cle/'

# train 데이터
for filename in os.listdir(TRAIN_PATH):
    
    try:
        if '.wav' not in filename:
            continue
         
        y,sr = librosa.load(TRAIN_PATH+filename)

        mfcc = librosa.feature.mfcc(y=y,sr=sr)
        mfcc_avg = []
        sum_ = 0 
        for i in mfcc:
            for j in i:
                sum_ += j
            mfcc_avg.append(sum_/10261)
        X_train.append(mfcc_avg)
            
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
    
# test 데이터
for filename in os.listdir(TEST_PATH):
    
    try:
        if '.wav' not in filename:
            continue
         
        y,sr = librosa.load(TEST_PATH+filename)

        mfcc = librosa.feature.mfcc(y=y,sr=sr)
        mfcc_avg = []
        sum_ = 0 
        for i in mfcc:
            for j in i:
                sum_ += j
            mfcc_avg.append(sum_/10261)
        X_test.append(mfcc_avg)
        
        if '_noise.wav' in filename:
            test_file_cnt+=1
            # test_name.append(filename[:11])
            y_test.append(noise)
        else:
            test_file_cnt+=1
            # test_name.append(filename[:11])
            y_test.append(cry)
        
    
    except Exception as e:
        print(filename,e)
        raise

print("==total number of train data files==")
print("the number of cry files:",total_number_of_train_cry)
print("the number of noise files:",total_number_of_train_noise)
print("==total train data length==")
print("total cry data length:",total_train_cry_data_len,"secs")
print("total noise data length:",total_train_noise_data_len,"secs")

# print("==test file name==")
# print(test_name)
print("the number of test files:",test_file_cnt)
print()

# # normalization
# scal=MinMaxScaler()

# scal.fit(X_train)
# X_train=scal.transform(X_train)
# X_test=scal.transform(X_test)

breakpoint()


# Logistic Regression model
clf = LogisticRegression(max_iter=3000)
clf.fit(X_train,y_train)
log_pred=clf.predict(X_test)
print("==logistic R eval==")
# print(classification_report(y_test,log_pred))
get_clf_eval(y_test,log_pred)
print()

# save model
logisticR_file="logisticR_model.pkl"
joblib.dump(clf,logisticR_file)

# SVM model
clf_svm=SVC()
clf_svm.fit(X_train,y_train)
svm_pred=clf_svm.predict(X_test)
print("==SVM eval==")
# print(classification_report(y_test,svm_pred))
get_clf_eval(y_test,svm_pred)
print()

# save model
svm_file="svm_model.pkl"
joblib.dump(clf_svm,svm_file)


# Linear Regression model
line_f=LinearRegression()
line_f.fit(X_train,y_train)
y_pred=line_f.predict(X_test)
y_b_pred=[]

for x in y_pred:
    if x<0.59:
        y_b_pred.append(0)
    else:
        y_b_pred.append(1)

y_b_pred=np.array(y_b_pred)

print("==Linear R eval==")
# print(classification_report(y_test,y_b_pred))
get_clf_eval(y_test,y_b_pred)
print()

#save model
linearR_file="linearR_model.pkl"
joblib.dump(line_f,linearR_file)


# mlp
mlp=MLPClassifier(max_iter=4000)
mlp.fit(X_train,y_train)
mlp_pred=mlp.predict(X_test)
print("==mlp eval==")
# print(classification_report(y_test,mlp_pred))
get_clf_eval(y_test,mlp_pred)
print()

# save model
mlp_file="mlp_model.pkl"
joblib.dump(mlp,mlp_file)

# knn
knn= KNeighborsClassifier(n_neighbors=100,n_jobs=-1)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
print("==knn eval==")
# print(classification_report(y_test,knn_pred))
get_clf_eval(y_test,knn_pred)
print()

# save model
knn_file="knn_model.pkl"
joblib.dump(knn,knn_file)

# lstm
# 
# 
# 
# 

# #save model
# lstm_file="lstm_model.pkl"    
# joblib.dump(lstm,lstm_file)


# loss function

# mse1 = mean_squared_error(y_test, log_pred)
# print('logistic mse = {:.3f}'.format(mse1))
# mae1 = mean_absolute_error(y_test, log_pred)
# print('logistic mae = {:.3f}'.format(mae1))

# mse2 = mean_squared_error(y_test, svm_pred)
# print('svm mse = {:.3f}'.format(mse2))
# mae2 = mean_absolute_error(y_test, svm_pred)
# print('svm mae = {:.3f}'.format(mae2))

# mse3 = mean_squared_error(y_test, mlp_pred)
# print('mlp mse = {:.3f}'.format(mse3))
# mae3 = mean_absolute_error(y_test, mlp_pred)
# print('mlp mae = {:.3f}'.format(mae3))


# mse4 = mean_squared_error(y_test, knn_pred)
# print('knn mse = {:.3f}'.format(mse4))
# mae4 = mean_absolute_error(y_test, knn_pred)
# print('knn mae = {:.3f}'.format(mae4))


# print("===y_predict compare===")
# print("y_test  : [",end='')
# for i in range(test_file_cnt):
#     # if i==37:
#     #     print()
#     if i!=test_file_cnt-1:
#         print(y_test[i],end=' ')
#     else:
#         print(y_test[i],end='')
# print("]")
# print("Logistic:",log_pred)
# print("SVM     :",svm_pred)
# # print("Linear R:",y_b_pred)
# print("mlp     :",mlp_pred)
# print("knn     :",knn_pred)





































