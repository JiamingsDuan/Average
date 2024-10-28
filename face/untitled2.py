# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:38:36 2019

@author: 楠
"""

from scipy.io import loadmat
import numpy as np
from sklearn import model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm as sk_svm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
image = loadmat("./ORL1024.mat")
image=image['allImages']
 #划分测试集训练集
 #u=np.mean(image, axis=1)    #求平均值    #注：求的是平均脸，既某位置400人脸的平均值，而不是人的平均值
 
#生成标签
label=[]
i=0
while i<40:
    for j in range(10):
        label.append(i)
    i=i+1
x=np.transpose(image)
y=label
 #划分测试集，训练集
x_train0, x_test0, y_train0, y_test0 =  model_selection.train_test_split( x, y, test_size=0.4)
def make_pca(i):
    u=np.mean(image, axis=1)    #求平均值    #注：求的是平均脸，既某位置400人脸的平均值，而不是人的平均值
    A=[]
    for t in range(1024):
      
        A.append(x_train0[:,t]-u[t])
      
    A=np.transpose(A)
    A=np.transpose(A)
     
    U,s,V=np.linalg.svd(A,full_matrices=0)   #full_matrices的取值是为0或者1，默认值为1，这时u的大小为(M,M)，
                                          #v的大小为(N,N) 。否则u的大小为(M,K)，v的大小为(K,N) ，K=min(M,N)。
    
    point=int(10.24*i)
    W=U[:,0:point]
    W=W.T
    P=np.dot(W,A)
    P=P.T
     
    B=[]
    for c in range(1024):
        B.append(x_test0[:,c]-u[c])
    B=np.transpose(B)
    B=np.transpose(B)
     
    Q=np.dot(W,B)
    Q=Q.T
    return Q,P

def knn_1024():
    get_score=[]
    for i in range(1,100):
# =============================================================================
#         pca = PCA(i,True,True)                             #建立pca类，设置参数，保留99%的数据方差
#         
#         P_pca = pca.fit_transform(x_train0)             #拟合并降维训练数据
#      
#         Q_pca = pca.transform(x_test0)
# =============================================================================
        Q_pca,P_pca=make_pca(i)
        clf=KNeighborsClassifier(n_neighbors=3)
           #进行训练
        clf.fit(P_pca,y_train0)
           #测试训练成绩
        get_score.append(clf.score(Q_pca,y_test0))
    plt.plot(get_score)
    arr_aa = np.array(get_score)
    maxindex  = np.argmax(arr_aa )
    n=int(10.24*(maxindex+1))
    print(n)
    return n
 
def svm():
    n=knn_1024()
    Q_pca,P_pca=make_pca(n)
    i=1
    acc_aa=[]
#    while i<20:
#        model = sk_svm.SVC(C=i, kernel='linear')
#        model.fit(P_pca,y_train0)
#        acc=model.score(Q_pca,y_test0) #根据给定数据与标签返回正确率的均值
#        acc_aa.append(acc)
#        print('SVM模型评价:',acc)
#        print(i)
#        i=i+1

#    plt.plot(acc_aa)
    lis=['rbf','linear','poly','sigmoid']
    for i in lis:
        model = sk_svm.SVC(C=1, kernel=i)
        model.fit(P_pca,y_train0)
        acc=model.score(Q_pca,y_test0) #根据给定数据与标签返回正确率的均值
        acc_aa.append(acc)
        print('SVM模型',i,'评价:',acc)
        plt.plot(acc_aa)
        plt.show()
if __name__ == '__main__':
    svm()
