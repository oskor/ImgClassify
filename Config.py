# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:17:00 2019

@author: JiaFenggang
"""
from BinTech.Utils.ConfigType import *
from BinTech.Utils.Augment import Augment,AugFunc,AugFuncTF
from BinTech.Model.ResnetClassify import ResnetSimpleClassify,resnet_v2_a3_mark


class Config:
    Training=False      #Train-True / Evalue-False
    Model_Save_Dir='./ckpt_tf2'
    Log_Dir='./ckpt_tf2'
    Pretrained_Mode=None
    #Pretrained_Mode='model.ckpt-10000'
    # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    #variables_to_restore=None

    if Training == False:
        Evalue_Model='./ckpt_tf2/step_1000.pb'
        Evalue_Data_Dir='../dataset/cifar-10-TFRecords/'
        Evalue_TFRecord_name =  r'cifar10_test.tfrecords'

    # Data
    Data_Dir='../dataset/cifar-10-TFRecords/' 
    Train_TFRecord_Name = r'cifar10_train.tfrecords' #by oskorlee
    Test_TFRecord_Name  = r'cifar10_test.tfrecords' #by oskorlee
    Precess_Func=None   # called on original data in both train and test stage
    # Augment_Func=Augment([1,3])    # called after Precess_Func in train stage
    # Augment_Func=AugFunc
    Augment_Func = AugFuncTF

    # Net
    Image_Size=[32,32,3]  #[h,w,c]
    Class_Num=10
    #Net=ResnetSimpleClassify(7,Class_Num)
    Net = resnet_v2_a3_mark(Class_Num)

    # Loss
    Loss=LossType.softmax_cross_entropy_with_logits

    # Optimizer
    Base_LR=3e-3 
    LRDecay=LRPolicyType.piecewise_constant_decay
    Optimizer=OptimizerType.AdamOptimizer # adam Base_LR=0.001

    # Train routine
    Epoch=1           # The maximum number of iterations
    Test_Iter=200        # Test iter num in testing (不再使用 by okorlee)
    Test_Interval=100   # Carry out testing every 500 training iterations. (不再使用 by okorlee)
    Log_Info_Snapshot = 100 # print log information every 100 training iterations by oskorlee
    Snapshot=500       # Snapshot intermediate results，Snapshot同时，(对验证即进行全部验证 by oskorlee)
    Batch_Size=50
    Shuffle_Buffer_Size = 10000 #by oskorlee

    

    
    
    
        
    
    
    