# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:40:50 2018

@author: Administrator
"""# -*- coding: UTF-8 -*-

import numpy as np
import os, glob, random
import scipy.io as sio 
import matplotlib.pyplot as plt
def pca(data, k):
    data = np.float32 (np.mat (data))
    
    rows, cols = data.shape  # 取大小
    data_mean = np.mean (data, 0)  # 求均值 平均脸
    Z = data - np.tile (data_mean, (rows, 1))  
    D, V = np.linalg.eig (Z * Z.T)  # 特征值与特征向量   Z * Z.T 协方差矩阵  实对称矩阵 特征脸
    V1 = V[:, :k]  # 取前k个特征向量 选择主成分
    V1 = Z.T * V1 
    
    for i in range (k):  # 特征向量归一化
        V1[:, i] /= np.linalg.norm (V1[:, i])
        
    return np.array (Z * V1), data_mean, V1   #训练集降低维度



def loadImageSet(sampleCount=8):  # 加载图像集，随机选择sampleCount张图片用于训练
    trainData = []
    testData = []
    yTrain = []
    yTest = []
    data =  sio.loadmat('ORL1024.mat')
    data=data['allImages']
    
    for k in range (40):
        sample = random.sample (range (10), sampleCount)
        trainData.extend ([data[:,(k*10)+i].ravel () for i in range (10) if i in sample])
        testData.extend ([data[:,(k*10)+i].ravel () for i in range (10) if i not in sample])
        yTest.extend ([k] * (10 - sampleCount))
        yTrain.extend ([k] * sampleCount)
        
        
    return np.array (trainData), np.array (yTrain), np.array (testData), np.array (yTest)



if __name__ == '__main__':
    
    
    xTrain_, yTrain, xTest_, yTest = loadImageSet ()
    num_train, num_test = xTrain_.shape[0], xTest_.shape[0]


    xTrain, data_mean, V = pca (xTrain_, 220)
    xTest = np.array ((xTest_ - np.tile (data_mean, (num_test, 1))) * V)  # 得到测试脸在特征向量下的数据
    yPredict = [yTrain[np.sum ((xTrain - np.tile (d, (num_train, 1))) ** 2, 1).argmin ()] for d in xTest]
    print ((yPredict == yTest).mean () * 100)