# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:54:32 2019

@author: JiaFenggang
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:52:37 2019

@author: JiaFenggang
"""

import tensorflow as tf
from Layers import Conv2D,FC

class Lenet_Config:
    Class_Num=10
    Image_Size=[32,32,3] #h,w,c
    Image_Dir='../dataset/cifar-10-python/cifar-10-batches-py/'
    Model_Save_Dir=''
    Log_Dir=''
    Batch_Size=32
    Epoch=5
    Lr=0.01
    
    

class Lenet:
    def __init__(self):
        self.conv_out_size=self.__conv_out_size(Lenet_Config.Image_Size[0],Lenet_Config.Image_Size[1])
        self.cls_num=Lenet_Config.Class_Num
    
    def __conv_out_size(self,h,w):
        f=lambda x: (((((((x-5)+1)-2)//2+1)-5)+1)-2)//2+1
        return f(h)*f(w)
    
    def __call__(self,imgs):
        in_channel=imgs.shape[-1].value
        conv_out_channel=16
        conv1_ret=Conv2D('conv1',imgs,[5,5,in_channel,6],stride=[1,1,1,1],padding='VALID',activate=True)
        max_pool1_ret=tf.nn.max_pool(conv1_ret,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='max_pool1')
        conv2_ret=Conv2D('conv2',max_pool1_ret,[5,5,6,conv_out_channel],stride=[1,1,1,1],padding='VALID',activate=True)
        max_pool2_ret=tf.nn.max_pool(conv2_ret,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='max_pool2')
        flatten=tf.reshape(max_pool2_ret,[-1,self.conv_out_size*conv_out_channel])
        fc1_ret=FC('fc1',flatten,120,activate=True)
        fc2_ret=FC('fc2',fc1_ret,84,activate=True)
        fc3_ret=FC('fc3',fc2_ret,self.cls_num)
        softmax_ret=tf.nn.softmax(fc3_ret,name='softmax')
        
        return softmax_ret
    
   