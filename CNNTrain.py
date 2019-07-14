# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:01:42 2019

@author: JiaFenggang
"""

import tensorflow as tf
import numpy as np
from Lenet import Lenet,Lenet_Config
from Cifar import CIFAR
from Dataset import DatasetNumpy
import os

class CNNTrain:
    def __init__(self,net,config):
        self.config=config
        self.log=self.__InitLogging(self.config.Log_Dir)
        self.log.info('Init CNNTrain')

        h,w,c=self.config.Image_Size        
        
        self.net=net
        
        self.imgs_=tf.placeholder(dtype=tf.float32,shape=[config.Batch_Size,h,w,c])
        self.labels_=tf.placeholder(dtype=tf.float32,shape=[config.Batch_Size,self.config.Class_Num])
        self.soft_max_ret=self.net(self.imgs_)
        self.loss=tf.reduce_mean(-tf.reduce_sum(self.labels_*tf.log(self.soft_max_ret),reduction_indices=[1]))
        correct_predict=tf.equal(tf.math.argmax(self.labels_,1),tf.math.argmax(self.soft_max_ret,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        self.train_step=tf.train.GradientDescentOptimizer(self.config.Lr).minimize(self.loss)
        
        # 查看所有变量的存储设备
        # sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess=tf.Session()
        # 检查是否所有变量都初始化了
        print(self.sess.run(tf.report_uninitialized_variables()))
        self.sess.run(tf.global_variables_initializer())
        print('确认全部初始化',self.sess.run(tf.report_uninitialized_variables()))
        
        
    def Train(self):
        sess=self.sess
        (train_image,train_label),(test_image,test_label),label_names=CIFAR(self.config.Image_Dir)
        train_ds=DatasetNumpy(train_image,train_label,self.config.Batch_Size)
        test_ds=DatasetNumpy(test_image,test_label,self.config.Batch_Size)
        
        for epoch in range(self.config.Epoch):
            train_ds.Shuffle()
            b_loss=self.__Train_epoch(sess,train_ds)
            test_ds.Shuffle()
            test_loss,test_acc=self.__Test_epoch(sess,test_ds)
            self.log.info('epoch'+str(epoch)+'\ttrain_loss='+str(b_loss)+'\ttest_loss='+str(test_loss)+'\ttest_acc='+str(test_acc))
        
        self.Save_model(sess)
        sess.close()
    
    def __Train_epoch(self,sess,train_ds):
        for b_img,b_label in train_ds:
            _,b_loss=sess.run((self.train_step,self.loss),feed_dict={self.imgs_:b_img,self.labels_:b_label})
        return b_loss
    
    def __Test_epoch(self,sess,test_ds):
        loss_li=[]
        acc_li=[]
        for b_img,b_label in test_ds:
            b_loss,b_acc=sess.run((self.loss,self.accuracy),feed_dict={self.imgs_:b_img,self.labels_:b_label})
            loss_li.append(b_loss)
            acc_li.append(b_acc)
    
        test_loss=np.average(np.array(loss_li))
        test_acc=np.average(np.array(acc_li))
        return test_loss,test_acc
            
    def Save_model(self,sess):
        save_name=self.config.Model_Save_Dir+'model'
        saver=tf.train.Saver()
        saver.save(sess,save_name)
    
    def Load_model(self,load_name):
        saver=tf.train.Saver()
        saver.restore(self.sess,load_name)
        
    
        
    def __InitLogging(self,log_file):
        import logging
        import datetime
        logger=logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter=logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
        # 输出到文件
        time_str=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')        
        fh=logging.FileHandler(os.path.join(log_file,time_str))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        # 输出到屏幕
        ch=logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        # logger中增加两个handler
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    
if __name__=='__main__':
    net=Lenet()
    train_app=CNNTrain(net,Lenet_Config)
    # Load_model一定在CNNTrain构造完成后调用
    # train_app.Load_model(os.path.join(Lenet_Config.Model_Save_Dir,'model'))
    train_app.Train()

        
        
        
        