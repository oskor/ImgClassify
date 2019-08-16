# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:24:03 2019

@author: JiaFenggang
"""

import numpy as np
from .Cifar import CIFAR10
import tensorflow as tf

'''
Use numpy.npArray data construct iterator
NOTE: This method is not suitable for large data sets
'''
class DatasetNumpy:
    '''
    parameters:
        np_img:shape[n,h,w,c]
        np_label:shape[n,class_num]
        batch_size:shape[1]
        aug_func:Augmentation method
    '''
    def __init__(self,np_img,np_label,batch_size):
        self.org_np_img=np_img
        self.org_np_label=np_label
        self.batch_size=batch_size
        
        self.len_img=np_img.shape[0]
        # 利用shuffle初始化self.cur_indices和self.index
        self.cur_indices=None
        self.index=0
        self.Shuffle()
        
    
    '''
    This function should be called at the beginning of each EPOCH.
    '''
    def Shuffle(self):
        self.cur_indices=np.arange(self.len_img)
        np.random.shuffle(self.cur_indices)
        # 防止最后一个batch不够，循环使用开头的
        self.cur_indices=np.concatenate([self.cur_indices,self.cur_indices[0:self.batch_size]])
        
        self.index=0
        self.batch_num=len(self.cur_indices)//self.batch_size
        
    
    def __iter__(self):
        return self
    
    # 1 epoch
    def __next__(self):
        try:
            # 触发边界异常，因为[a:b]不会触发边界异常
            self.cur_indices[self.index+self.batch_size]
            idx=self.cur_indices[self.index:self.index+self.batch_size]
            img=self.org_np_img[idx]
            label=self.org_np_label[idx]        
        except IndexError:
            raise StopIteration()
        self.index+=self.batch_size
        return img,label


    def __getitem__(self,batch_index):
        img_idx=batch_index*self.batch_size
        if img_idx+self.batch_size>=len(self.cur_indices):
            self.Shuffle()
        img_idx=(img_idx%self.batch_num)*self.batch_size
        idx=self.cur_indices[img_idx:img_idx+self.batch_size]
        img=self.org_np_img[idx]
        label=self.org_np_label[idx]    

        return img,label


    def __len__(self):
        return self.org_np_img.shape[0]

class ClassDatasetTFRecord(object):
    def __init__(self,im_width,im_height,im_channel,class_num,epoch,batch_size,is_shuffle=False,shuffle_buffer_size=None,augment_func=None,is_one_iter=True):
        self.im_width    = im_width
        self.im_height   = im_height
        self.im_channel  = im_channel
        self.class_num   = class_num
        self.epoch       = epoch
        self.batch_size  = batch_size
        self.is_shuffle  = is_shuffle
        self.augment_func= augment_func
        self.is_one_iter = is_one_iter
        self.dataset     = None

        if shuffle_buffer_size is None:
            self.shuffle_buffer_size = 4*batch_size
        else:
            self.shuffle_buffer_size = shuffle_buffer_size
        
    
    def __parse_proto(self,example_proto):
        features = tf.parse_single_example(example_proto,features=
            {
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.float32),
            'name' :tf.FixedLenFeature([],tf.string)
            })

        im_dim = tf.convert_to_tensor([self.im_channel,self.im_height,self.im_width])
        image = tf.transpose(tf.reshape(tf.decode_raw(features['image'],tf.uint8),im_dim),[1,2,0])

        if self.augment_func is not None:
            image = self.augment_func(image)
        
        label = tf.cast(tf.convert_to_tensor(features['label']),tf.int32)
        label = tf.one_hot(label,tf.convert_to_tensor(self.class_num), 1, 0) 
        image = tf.cast(image,tf.float32)
        label = tf.cast(label,tf.float32)

        return image,label

    def __call__(self,TFRecord_filename):
        return self.trf_batch_data(TFRecord_filename)

    def trf_batch_data(self,TFRecord_filename):

        self.trf_batch_iterator(TFRecord_filename)
        image_batch,label_batch = self.iterator.get_next()

        return image_batch,label_batch
    
    def trf_batch_iterator(self,TFRecord_filename):

        if not isinstance(TFRecord_filename,list):
            TFRecord_filename = [TFRecord_filename]
        self.dataset = tf.data.TFRecordDataset(TFRecord_filename)
        self.dataset = self.dataset.map(self.__parse_proto,num_parallel_calls=8).repeat(self.epoch)
        if self.is_shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        self.dataset = self.dataset.prefetch(self.batch_size).batch(self.batch_size)

        if self.is_one_iter:
            self.iterator = self.dataset.make_one_shot_iterator()
        else:
            self.iterator = self.dataset.make_initializable_iterator()
   
    def iterator_initialize(self,sess):
        if self.is_one_iter is False:
            sess.run(self.iterator.initializer)
        else:
            print('The iterator of this dataset is one shot which can not be initlialized! ')



class ClassDatasetTFRecord_v2(object):
    def __init__(self,im_width,im_height,im_channel,class_num,epoch,batch_size,is_shuffle=False,shuffle_buffer_size=None,augment_func=None):
            self.im_width    = im_width
            self.im_height   = im_height
            self.im_channel  = im_channel
            self.class_num   = class_num
            self.epoch       = epoch
            self.batch_size  = batch_size
            self.is_shuffle  = is_shuffle
            self.augment_func= augment_func
            self.dataset     = None
    
            if shuffle_buffer_size is None:
                self.shuffle_buffer_size = 4*batch_size
            else:
                self.shuffle_buffer_size = shuffle_buffer_size
            
    def __parse_proto(self,example_proto):
        features = tf.parse_single_example(example_proto,features=
            {
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.float32),
            'name' :tf.FixedLenFeature([],tf.string)
            })

        im_dim = tf.convert_to_tensor([self.im_channel,self.im_height,self.im_width])
        image = tf.transpose(tf.reshape(tf.decode_raw(features['image'],tf.uint8),im_dim),[1,2,0])

        if self.augment_func is not None:
            image = self.augment_func(image)
        
        label = tf.cast(tf.convert_to_tensor(features['label']),tf.int32)
        label = tf.one_hot(label,tf.convert_to_tensor(self.class_num), 1, 0) 
        image = tf.cast(image,tf.float32)
        image = tf.multiply(image,1/255.0)
        label = tf.cast(label,tf.float32)

        return image,label

    def __call__(self,TFRecord_filename):
        if not isinstance(TFRecord_filename,list):
            TFRecord_filename = [TFRecord_filename]
        self.dataset = tf.data.TFRecordDataset(TFRecord_filename)
        self.dataset = self.dataset.map(self.__parse_proto,num_parallel_calls=8).repeat(self.epoch)
        if self.is_shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        self.dataset = self.dataset.prefetch(self.batch_size).batch(self.batch_size)

        iterator = self.dataset.make_initializable_iterator()

        return iterator,self.dataset.output_types,self.dataset.output_shapes

if __name__=='__main__':
    (train_img,train_label),(_,_),_=CIFAR10('../dataset/cifar-10-python/cifar-10-batches-py/')
    ds=DatasetNumpy(train_img,train_label,7)
    print(len(ds))
    print(type(ds[3]))
    