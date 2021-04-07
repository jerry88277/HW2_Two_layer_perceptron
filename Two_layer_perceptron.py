# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:43:02 2021

@author: JerryDai
"""
import numpy as np
import pandas as pd
from numpy.random import randn
import math
import cv2
import time
import matplotlib.pyplot as plt


def load_pic_data_by_txt(pic_load_list): #pic_load_list = tmp_load_list
    tmp_hist_df = []
    for index, tmp_pic_path in enumerate(pic_load_list):
        # print(index)
        # print(tmp_pic_path)
        temp_pic = cv2.imread(tmp_pic_path)
        colors = ('b', 'g', 'r')
        
        tmp_hist_array = np.array([])
        for i, col in enumerate(colors):
            hist = cv2.calcHist([temp_pic], [i], None, [256], [0, 256])
            hist = hist.flatten()
            tmp_hist_array = np.append(tmp_hist_array, hist)
        if index == 0:
            tmp_hist_df = pd.DataFrame(tmp_hist_array).T
        else:
            tmp_hist_df = tmp_hist_df.append(pd.DataFrame(tmp_hist_array).T)
    tmp_train_x = tmp_hist_df.reset_index(drop = True)
    return tmp_train_x

def model_fit(x, w1, w2):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    return y_pred 

def softmax(y_pred):
    for row in range(len(y_pred)):
    # print(row)
        y_pred.iloc[row, :] = np.exp(y_pred.iloc[row, :]) / np.sum(np.exp(y_pred.iloc[row, :]))
    return y_pred

def cross_entropy(y_pred, y):
    loss = 0
    for index, label in enumerate(y):
        # print(index, " & ", label)
        loss += np.log2(y_pred.iloc[index, label])
    return (-1)*(loss)

def gradient(y_pred, y):
    grad_y_pred = y_pred
    for index, label in enumerate(y):
        # print(index, " & ", label)
        grad_y_pred.iloc[index, label] -= 1
    return grad_y_pred

def get_accuracy(data_list, batch_size, w1, w2): # data_list = val_txt   data_list = test_txt
    correct = 0
    total = len(data_list)
    for j in range(math.ceil(len(data_list) / batch_size)):
        start_point = i * batch_size
        end_point = (i + 1) * batch_size
        
        if end_point > len(data_list):
            end_point = len(data_list)
        
        tmp_load_list = list(data_list.loc[start_point : end_point - 1, 'pic_path'])
        x = load_pic_data_by_txt(tmp_load_list)
        y = data_list.loc[start_point : end_point - 1, 'label']
    
        y_pred = softmax(model_fit(x, w1, w2))
    
        for index, label in enumerate(y):
            if np.argmax(y_pred.iloc[index, :]) == label:
                correct += 1
    
    accuracy = correct / total
    return accuracy
    
## Settings
batch_size = 32
epochs = 1
learning_rate = [0.01, 0.01, 0.01]
N, D_in, H, D_out = batch_size, 768, 256, 50
# x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)
train_acc, val_acc, test_acc = [], [], []

## Settings
train_txt = pd.read_csv('train.txt', header = None, names = ['pic_path', 'label'], sep = ' ')

val_txt = pd.read_csv('val.txt', header = None, names = ['pic_path', 'label'], sep = ' ')
# val_x = load_pic_data_by_txt(val_txt['pic_path'])
# val_y = val_txt['label']

test_txt = pd.read_csv('test.txt', header = None, names = ['pic_path', 'label'], sep = ' ') 
# test_x = load_pic_data_by_txt(test_txt['pic_path'])
# test_y = test_txt['label']
    
## train model
loss_info = []
start = time.time()
for i in range(epochs):
    train_txt_shuffled = train_txt.sample(frac = 1).reset_index(drop = True)
    # Load Data by batch
    for j in range(math.ceil(len(train_txt_shuffled) / batch_size)):
        start_point = i * batch_size
        end_point = (i + 1) * batch_size
        # print(start_point)
        # print(end_point)
        
        if end_point > len(train_txt_shuffled):
            end_point = len(train_txt_shuffled)
        
        tmp_load_list = list(train_txt_shuffled.loc[start_point : end_point - 1, 'pic_path'])
        train_x = load_pic_data_by_txt(tmp_load_list) # train_x = load_pic_data_by_txt(train_txt['pic_path'])
        train_y = train_txt_shuffled.loc[start_point : end_point - 1, 'label']
    
        # predict
        h = 1 / (1 + np.exp(-train_x.dot(w1)))
        y_pred = h.dot(w2)
        
        # softmax
        y_pred = softmax(y_pred)
        
        # Cross-entropy loss
        loss = cross_entropy(y_pred, train_y)
        print('epoch:', i, 'iter:', j, ',loss:', loss)
        loss_info.append(loss)
        
        # get accuracy
        if j % 50 == 0:
            train_acc.append(get_accuracy(train_txt, batch_size, w1, w2))
            val_acc.append(get_accuracy(val_txt, batch_size, w1, w2))
            test_acc.append(get_accuracy(test_txt, batch_size, w1, w2))
            
        # Update gradients
        grad_y_pred = gradient(y_pred, train_y)
        
        grad_w2 = h.T.dot(grad_y_pred)
        grad_h = grad_y_pred.dot(w2.T)
        grad_w1 = train_x.T.dot(grad_h * h * (1 - h))
        
        # Update weights
        w1 -= learning_rate[i] * grad_w1
        w2 -= learning_rate[i] * grad_w2
        
end = time.time()
print("Training time：%f sec" % (end - start))

plt.plot(loss_info)
plt.title('Triaining Curve(batch size = 32, lr = 0.01)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('Training_Curve.png')
plt.close()

plt.plot(train_acc, label = 'train')
plt.plot(val_acc, label = 'val')
plt.plot(test_acc, label = 'test')
plt.title('Accuracy')
plt.legend()
plt.savefig('Accuracy.png')
plt.close()
