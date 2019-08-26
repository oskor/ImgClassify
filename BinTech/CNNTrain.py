# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:01:42 2019

@author: JiaFenggang
"""

import tensorflow as tf
import numpy as np
from .Utils.ConfigParser import *
from .DataSet.Cifar import CIFAR10
from .DataSet.Dataset import DatasetNumpy,ClassDatasetTFRecord,ClassDatasetTFRecord_v2
from tensorflow.python.framework import graph_util

import os

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.9
tfconfig.gpu_options.allow_growth = True

class CNNTrain:
    def __init__(self,config):
        self.config=config
        self.log=self.__InitLogging(self.config.Log_Dir)
        self.log.info('Init CNNTrain')

        h,w,c=self.config.Image_Size        
        
        self.imgs_=tf.placeholder(dtype=tf.float32,shape=[None,h,w,c],name='imgs')
        self.labels_=tf.placeholder(dtype=tf.float32,shape=[None,self.config.Class_Num],name='labels')
        self.train_mode_=tf.placeholder(dtype=tf.bool,name='training')
        self.keep_prob_=tf.placeholder(tf.float32,name='keep_prob')
        # record current iter num
        global_steps=tf.Variable(0, name='global_step', trainable=False)

        net_ret=self.config.Net(self.imgs_,self.train_mode_,self.keep_prob_)
        # 
        self.loss=get_loss(self.config,self.labels_,net_ret)
        self.softmax_out = tf.nn.softmax(net_ret, name = 'output_node')
        correct_predict=tf.equal(tf.math.argmax(self.labels_,1),tf.math.argmax(self.softmax_out,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32),name='acc')

        self.lr_rate=get_LRPolicy(self.config,global_steps)
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step=get_optimizer(self.config,self.lr_rate).minimize(self.loss,global_step=global_steps)

        if not os.path.isdir(self.config.Model_Save_Dir):
            os.mkdir(self.config.Model_Save_Dir)
        
        self.saver=tf.train.Saver(max_to_keep=0)
        self.output_nodes = ['acc', 'output_node']
        self.sess=tf.Session(config=tfconfig) 
        # 检查是否所有变量都初始化了
        print(self.sess.run(tf.report_uninitialized_variables()))
        self.sess.run(tf.global_variables_initializer())
        print('确认全部初始化',self.sess.run(tf.report_uninitialized_variables()))

        if self.config.Pretrained_Mode is not None:
            re,self.step = self.Load_model(self.config.Pretrained_Mode)
            if re is False:
                self.step = 0
        else:
            self.step = 0

    def Train(self):
       
        sess = self.sess
        
        (train_image,train_label),(test_image,test_label),_=CIFAR10(self.config.Data_Dir)
        train_ds=DatasetNumpy(train_image,train_label,self.config.Batch_Size)
        test_ds=DatasetNumpy(test_image,test_label,self.config.Batch_Size)
                
        max_iter = int(len(train_ds) / self.config.Batch_Size * self.config.Epoch)
        #max_iter=1000
        for step in range(self.step,max_iter):
            b_train_imgs,b_train_labels=train_ds[step]
            # precess
            if None != self.config.Precess_Func:
                b_train_imgs=self.config.Precess_Func(b_train_imgs)
            # data augment
            if None != self.config.Augment_Func:
                b_train_imgs=self.config.Augment_Func(b_train_imgs)

            _,b_train_loss,b_train_acc=sess.run((self.train_step,self.loss,self.accuracy),feed_dict=\
            {self.imgs_:b_train_imgs,self.labels_:b_train_labels,self.keep_prob_: 0.5, self.train_mode_: True})

            if((step+1) % self.config.Test_Interval)==0:
                test_loss_li=[]
                test_acc_li=[]
                test_ds.Shuffle()
                for test_step in range(self.config.Test_Iter):
                    b_test_imgs,b_test_labels=test_ds[test_step]
                    # precess
                    if None != self.config.Precess_Func:
                        b_test_imgs=self.config.Precess_Func(b_test_imgs)
                    b_test_loss,b_test_acc=sess.run((self.loss,self.accuracy),feed_dict=\
                        {self.imgs_:b_test_imgs,self.labels_:b_test_labels,self.keep_prob_: 1, self.train_mode_: False})
                    test_loss_li.append(b_test_loss)
                    test_acc_li.append(b_test_acc)
                test_loss=np.average(np.array(test_loss_li))
                test_acc=np.average(np.array(test_acc_li))
                self.log.info('step:'+str(step+1)+'\ttrain_loss='+str(b_train_loss)+'\ttrain_acc='+str(b_train_acc)+'\ttest_loss='+str(test_loss)+'\ttest_acc='+str(test_acc))
            
            if((step+1) % self.config.Snapshot)==0:
                self._SaveCheckPoint(sess,step+1)

        sess.close()
    
    def _SaveCheckPoint(self,sess,step):
        self.Save_model(sess,step)
        constant_graph = graph_util.convert_variables_to_constants(sess=sess,
                                                                input_graph_def=sess.graph_def,
                                                                output_node_names=self.output_nodes)
        # for i,n in enumerate(self.constant_graph.node):
        #     print('Name of the node - %s' % n.name)
        model_name=os.path.join(self.config.Model_Save_Dir,'step_'+str(step)+'.pb')
        with tf.gfile.FastGFile(model_name, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    def __InitLogging(self,log_file):
        import logging
        import datetime

        if not os.path.isdir(log_file):
            os.mkdir(log_file)

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

    def Save_model(self,sess,step):
        save_name=os.path.join(self.config.Model_Save_Dir,'model.ckpt')
        self.saver.save(sess,save_name, global_step=step)
    
    def Load_model(self,load_name):
        import re
        try:
            self.saver.restore(self.sess, os.path.join(self.config.Model_Save_Dir, load_name))
            step = int(next(re.finditer("(\d+)(?!.*\d)", load_name)).group(0))
            print(" [*] Success to read {}".format(load_name))
            return True,step
        except:
            return False,0


class CNNTrain_v1(CNNTrain):
    def __init__(self,config):
        super(CNNTrain_v1,self).__init__(config)
    
    def Train(self):

        with tf.device("/cpu:0"):
            h,w,c=self.config.Image_Size 

            dataset_train = ClassDatasetTFRecord(w,h,c,\
            self.config.Class_Num,\
            self.config.Epoch,\
            self.config.Batch_Size,\
            True,\
            self.config.Shuffle_Buffer_Size,\
            self.config.Augment_Func)
            dataset_test = ClassDatasetTFRecord(w,h,c,\
            self.config.Class_Num,\
            1,\
            self.config.Batch_Size,\
            False,\
            is_one_iter = False)

            b_image_train,b_label_train = dataset_train(os.path.join(self.config.Data_Dir,self.config.Train_TFRecord_Name))
            b_image_test,b_label_test   = dataset_test(os.path.join(self.config.Data_Dir,self.config.Test_TFRecord_Name))

        sess = self.sess
        step = self.step

        try:
            while True:
                b_train_imgs,b_train_labels=sess.run([b_image_train,b_label_train])
                _,b_train_loss,b_train_acc=sess.run((self.train_step,self.loss,self.accuracy),feed_dict=\
                {self.imgs_:b_train_imgs,self.labels_:b_train_labels,self.keep_prob_: 0.5, self.train_mode_: True})
                
                lr_rate_np = sess.run(self.lr_rate)
                if ((step+1) % self.config.Log_Info_Snapshot)==0:
                    self.log.info('step:'+str(step+1)+'\tlr_rate='+str(lr_rate_np)+'\ttrain_loss='+str(b_train_loss)+'\ttrain_acc='+str(b_train_acc))
                
                if((step+1) % self.config.Snapshot)==0:
                    test_loss_li=[]
                    test_acc_li=[]
                    dataset_test.iterator_initialize(sess)
                    try:
                        while True:
                            b_test_imgs,b_test_labels=sess.run([b_image_test,b_label_test])
                            b_test_loss,b_test_acc=sess.run((self.loss,self.accuracy),feed_dict=\
                                {self.imgs_:b_test_imgs,self.labels_:b_test_labels,self.keep_prob_: 1, self.train_mode_: False})
                            test_loss_li.append(b_test_loss)
                            test_acc_li.append(b_test_acc)
                    except tf.errors.OutOfRangeError:
                        pass
                    test_loss=np.average(np.array(test_loss_li))
                    test_acc=np.average(np.array(test_acc_li))

                    self.log.info('step:'+str(step+1)+'\ttest_loss='+str(test_loss)+'\ttest_acc='+str(test_acc))
                    self._SaveCheckPoint(sess,step+1)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Train Finished!!!')
        self._SaveCheckPoint(sess,step+1)
        sess.close()

class CNNTrain_v2(object):
    def __init__(self,config):
        self.config=config
        self.log=self.__InitLogging(self.config.Log_Dir)
        self.log.info('Init CNNTrain')

        #with tf.device("/gpu:0"):

        h,w,c=self.config.Image_Size 
        dataset_train = ClassDatasetTFRecord_v2(w,h,c,\
        self.config.Class_Num,\
        self.config.Epoch,\
        self.config.Batch_Size,\
        True,\
        self.config.Shuffle_Buffer_Size,\
        self.config.Augment_Func)
        dataset_test= ClassDatasetTFRecord_v2(w,h,c,\
        self.config.Class_Num,\
        1,\
        self.config.Batch_Size,\
        False)

        self.train_iter,train_output_type,train_output_shape = dataset_train(os.path.join(self.config.Data_Dir,self.config.Train_TFRecord_Name))
        self.test_iter,_,_ = dataset_test(os.path.join(self.config.Data_Dir,self.config.Test_TFRecord_Name))

        self.iter_handle = tf.placeholder(tf.string, shape=[],name='iter_handle')
        self.train_mode_=tf.placeholder(dtype=tf.bool,name='training')
        self.keep_prob_=tf.placeholder(tf.float32,name='keep_prob')
        # record current iter num
        global_steps=tf.Variable(0, name='global_step', trainable=False)

        # net graph
        iterator = tf.data.Iterator.from_string_handle(self.iter_handle,train_output_type,train_output_shape)
        self.imgs_, self.labels_ = iterator.get_next()
        net_ret=self.config.Net(self.imgs_,self.train_mode_,self.keep_prob_)
        #net_ret=self.config.Net(self.imgs_)
        # 
        self.loss=get_loss(self.config,self.labels_,net_ret)
        self.softmax_out = tf.nn.softmax(net_ret, name = 'output_node')
        correct_predict=tf.equal(tf.math.argmax(self.labels_,1),tf.math.argmax(self.softmax_out,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32),name='acc')

        self.lr_rate=get_LRPolicy(self.config,global_steps)
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step=get_optimizer(self.config,self.lr_rate).minimize(self.loss,global_step=global_steps)

        if not os.path.isdir(self.config.Model_Save_Dir):
            os.mkdir(self.config.Model_Save_Dir)
        
        self.saver=tf.train.Saver(max_to_keep=0)
        self.output_nodes = ['acc', 'output_node']
        self.sess=tf.Session(config=tfconfig)
        print(self.sess.run(tf.report_uninitialized_variables()))
        self.sess.run(tf.global_variables_initializer())
        print('确认全部初始化',self.sess.run(tf.report_uninitialized_variables()))

        if self.config.Pretrained_Mode is not None:
            re,self.step = self.Load_model(self.config.Pretrained_Mode)
            if re is False:
                self.step = 0
        else:
            self.step = 0
    
    def Train(self):
        sess = self.sess

        handle_train,handle_test = sess.run([self.train_iter.string_handle(),self.test_iter.string_handle()])

        sess.run(self.train_iter.initializer)
        step =self.step
        try:
            while True:
                _,b_train_loss,b_train_acc=sess.run((self.train_step,self.loss,self.accuracy),feed_dict=\
                {self.iter_handle:handle_train,self.keep_prob_: 0.5, self.train_mode_: True})
                lr_rate_np = sess.run(self.lr_rate)
                if ((step+1) % self.config.Log_Info_Snapshot)==0:
                    self.log.info('step:'+str(step+1)+'\tlr_rate='+str(lr_rate_np)+'\ttrain_loss='+str(b_train_loss)+'\ttrain_acc='+str(b_train_acc))
                
                if((step+1) % self.config.Snapshot)==0:
                    test_loss_li=[]
                    test_acc_li=[]
                    sess.run(self.test_iter.initializer)
                    try:
                        while True:
                            b_test_loss,b_test_acc=sess.run((self.loss,self.accuracy),feed_dict=\
                                {self.iter_handle:handle_test,self.keep_prob_: 1.0, self.train_mode_: False})
                            test_loss_li.append(b_test_loss)
                            test_acc_li.append(b_test_acc)
                    except tf.errors.OutOfRangeError:
                        pass
                    
                    test_loss=np.average(np.array(test_loss_li))
                    test_acc=np.average(np.array(test_acc_li))
                
                    self.log.info('step:'+str(step+1)+'\ttest_loss='+str(test_loss)+'\ttest_acc='+str(test_acc))
                    self._SaveCheckPoint(sess,step+1)    
                step += 1
        except tf.errors.OutOfRangeError:
            print('Train Finished!!!')
        self._SaveCheckPoint(sess,step+1)
        sess.close()

    def __InitLogging(self,log_file):
        import logging
        import datetime

        if not os.path.isdir(log_file):
            os.mkdir(log_file)

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

    def _SaveCheckPoint(self,sess,step):
        self.Save_model(sess,step)
        constant_graph = graph_util.convert_variables_to_constants(sess=sess,
                                                                input_graph_def=sess.graph_def,
                                                                output_node_names=self.output_nodes)
        #for i,n in enumerate(constant_graph.node):
             #print('Name of the node - %s' % n.name)
        model_name=os.path.join(self.config.Model_Save_Dir,'step_'+str(step)+'.pb')
        with tf.gfile.FastGFile(model_name, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    def Save_model(self,sess,step):
        save_name=os.path.join(self.config.Model_Save_Dir,'model.ckpt')
        self.saver.save(sess,save_name, global_step=step)
    
    def Load_model(self,load_name):
        import re
        try:
            self.saver.restore(self.sess, os.path.join(self.config.Model_Save_Dir, load_name))
            step = int(next(re.finditer("(\d+)(?!.*\d)", load_name)).group(0))
            print(" [*] Success to read {}".format(load_name))
            return True,step
        except:
            return False,0

        