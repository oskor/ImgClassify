# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 00:07:34 2019

@author: JiaFenggang
"""
import tensorflow as tf
'''
parameters:
    input-shape[b,h,w,c]
    kernel-shape[kh,kw,in,out]
    activate-use activate function or NOT, default relu
'''
def Conv2D(name,input,kernel,stride,padding,activate=True):
    n_out=kernel[-1]
    # 注意tf.name_scope不能影响到variable，命名还是加上scope，比如:scope+'w'
    # 也可以使用tf.variable_scope，看习惯
    with tf.name_scope(name) as scope:
        w=tf.get_variable(scope+'w',
                          shape=kernel,
                          dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b=tf.get_variable(scope+'b',
                          initializer=tf.constant(0.0,shape=[n_out],dtype=tf.float32))
        conv_ret=tf.nn.bias_add(tf.nn.conv2d(input,w,stride,padding),b)
        if activate:
            conv_ret=tf.nn.relu(conv_ret)
            
        return conv_ret

'''
parameters:
    input-shape[b,m]
    n_out-shape[b,n_out]
    activate-use activate function or NOT, default relu
'''

def FC(name,input,n_out,activate=True):
    n_in=input.shape[-1].value
    with tf.name_scope(name) as scope:
        w=tf.get_variable(scope+'w',
                          shape=[n_in,n_out],
                          dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
        b=tf.get_variable(scope+'b',
                          initializer=tf.constant(0.0,shape=[n_out],dtype=tf.float32))
        fc_ret=tf.matmul(input,w)+b
        if activate:
            fc_ret=tf.nn.relu(fc_ret)
            
        return fc_ret

       
#TODO 需要增加
#def CBR():
