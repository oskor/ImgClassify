# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:10:38 2019

@author: JiaFenggang
"""
import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot(label,class_num):
    label_one_hot=(np.arange(class_num)==label[:,None]).astype(np.float32)
    return label_one_hot


'''
parameters:
    file_dir: CIFAR10(100)数据所在文件夹
return:
    x_train:shape[n,h,w,c]
    y_train:shape[n,10]
    x_test:shape[n,h,w,c]
    y_test:shape[n,10]
    y_name:shape[n]

'''
def CIFAR(file_dir):
    # load train data
    x_train_li=[]
    y_train_li=[]
    # data_batch_1...data_batch_5
    for i in range(1,6):
        cur_dic=unpickle(os.path.join(file_dir,'data_batch_'+str(i)))
        x_train_li.append(cur_dic[b'data'])
        y_train_li.append(cur_dic[b'labels'])
    x_train=np.reshape(np.concatenate(x_train_li),(50000,3,32,32)).transpose(0,2,3,1).astype(np.float32)/255.0
    y_train=np.reshape(np.concatenate(y_train_li),(50000))
    y_train=one_hot(y_train,10)
    # load test data test_batch
    test_dic=unpickle(os.path.join(file_dir,'test_batch'))
    x_test=np.reshape(test_dic[b'data'],(10000,3,32,32)).transpose(0,2,3,1).astype(np.float32)/255.0
    y_test=np.reshape(test_dic[b'labels'],(10000))
    y_test=one_hot(y_test,10)
    # load label name
    name_dict=unpickle(os.path.join(file_dir,'batches.meta'))
    y_names=name_dict[b'label_names']
    
    return (x_train,y_train),(x_test,y_test),y_names
    #num=20
    #return (x_train[:num],y_train[:num]),(x_test[:num],y_test[:num]),y_names