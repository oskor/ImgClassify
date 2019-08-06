# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:17:00 2019

@author: JiaFenggang
"""
from ConfigType import *
from Augment import Augment,AugFunc
from Resnet import ResnetCifar10
from Model.ResnetClassify import    ResnetSimpleClassify


class Config:
    Training=False      #Train-True / Evalue-False
    Model_Save_Dir=''
    Log_Dir=''
    Pretrained_Mode=None
    Pretrained_Mode='model1000'
    # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    variables_to_restore=None

    if Training == False:
        Evalue_Model='step_1000.pb'
        Evalue_Data_Dir='../dataset/cifar-10-python/cifar-10-batches-py/' 

    # Data
    Data_Dir='../dataset/cifar-10-python/cifar-10-batches-py/' 
    Precess_Func=None   # called on original data in both train and test stage
    # Augment_Func=Augment([1,3])    # called after Precess_Func in train stage
    Augment_Func=AugFunc

    # Net
    Image_Size=[32,32,3]  #[h,w,c]
    Class_Num=10
    Net=ResnetCifar10(9)
    # Net=ResnetSimpleClassify(9,Class_Num)

    # Loss
    Loss=LossType.softmax_cross_entropy_with_logits

    # Optimizer
    Base_LR=0.00001  
    LRDecay=LRPolicyType.piecewise_constant_decay
    Optimizer=OptimizerType.AdamOptimizer # adam Base_LR=0.001

    # Train routine
    Epoch=1           # The maximum number of iterations
    Test_Iter=100        # Test iter num in testing
    Test_Interval=100   # Carry out testing every 500 training iterations.
    Snapshot=500       # Snapshot intermediate results
    Batch_Size=32

    

    
    
    
        
    
    
    