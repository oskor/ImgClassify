# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:24:03 2019

@author: JiaFenggang
"""

import numpy as np
from Cifar import CIFAR

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
    def __init__(self,np_img,np_label,batch_size,aug_func=None):
        self.org_np_img=np_img
        self.org_np_label=np_label
        self.batch_size=batch_size
        self.aug_func=aug_func
        
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
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            # 触发边界异常，因为[a:b]不会触发边界异常
            self.cur_indices[self.index+self.batch_size]
            idx=self.cur_indices[self.index:self.index+self.batch_size]
            img=self.org_np_img[idx]
            if self.aug_func != None:
                img=self.aug_func(img)
            label=self.org_np_label[idx]        
        except IndexError:
            raise StopIteration()
        self.index+=self.batch_size
        return img,label


if __name__=='__main__':
    (train_img,train_label),(_,_),_=CIFAR('../dataset/cifar-10-python/cifar-10-batches-py/')
    ds=DatasetNumpy(train_img,train_label,7)
    for epoch in range(10):
        ds.Shuffle()
        for img,label in ds:
            print('i_shape\t',img.shape,'\tl_shape\t',label.shape)
    