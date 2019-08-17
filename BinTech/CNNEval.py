import tensorflow as tf
from .DataSet.Cifar import CIFAR10
from .DataSet.Dataset import DatasetNumpy,ClassDatasetTFRecord,ClassDatasetTFRecord_v2
from tensorflow.python.framework import graph_util
import numpy as np

import os

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.9
tfconfig.gpu_options.allow_growth = True

def Eval(model_name,data_dir,batch_size=32):
    with open(model_name,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
    graph=tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def,name='')
        for i,n in enumerate(graph_def.node):
            print('Name of the node- %s' % n.name)
        imgs_=graph.get_tensor_by_name('imgs:0')
        labels_=graph.get_tensor_by_name('labels:0')
        train_mode_=graph.get_tensor_by_name('training:0')
        keep_prob_=graph.get_tensor_by_name('keep_prob:0')
        acc_op=graph.get_tensor_by_name('acc:0')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            (_,_),(test_image,test_label),_=CIFAR10(data_dir)
            test_ds=DatasetNumpy(test_image,test_label,batch_size)
            acc_li=[]
            for b_test_imgs,b_test_labels in test_ds:
                b_acc=sess.run(acc_op,feed_dict={imgs_:b_test_imgs,labels_:b_test_labels,keep_prob_:1,train_mode_:False})
                acc_li.append(b_acc)
            test_acc=np.average(np.array(acc_li))
            print('test accuracy: '+str(test_acc))

def Eval_v1(model_name,TFRecord_file_path,image_size,class_num,batch_size):
    with open(model_name,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
    graph=tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def,name='')
        for i,n in enumerate(graph_def.node):
            print('Name of the node- %s' % n.name)

        imgs_=graph.get_tensor_by_name('imgs:0')
        labels_=graph.get_tensor_by_name('labels:0')
        train_mode_=graph.get_tensor_by_name('training:0')
        keep_prob_=graph.get_tensor_by_name('keep_prob:0')
        acc_op=graph.get_tensor_by_name('acc:0')
        
        with tf.device("/cpu:0"):
            h,w,c = image_size
            dataset_test = ClassDatasetTFRecord(w,h,c,class_num,1,batch_size,False)
            b_image_test,b_label_test = dataset_test(TFRecord_file_path)

        with tf.Session(config=tfconfig) as sess:
            sess.run(tf.global_variables_initializer())     
            acc_li=[]
            try:
                while True:
                    b_test_imgs,b_test_labels=sess.run([b_image_test,b_label_test])
                    b_acc=sess.run(acc_op,feed_dict={imgs_:b_test_imgs,labels_:b_test_labels,keep_prob_:1,train_mode_:False})
                    acc_li.append(b_acc)
            except tf.errors.OutOfRangeError:
                pass
            test_acc=np.average(np.array(acc_li))
            print('test accuracy: '+str(test_acc))

def  Eval_v2(model_name,TFRecord_filepath,image_size,class_num,batch_size):
    with open(model_name,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
    graph=tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def,name='')
        for i,n in enumerate(graph_def.node):
            print('Name of the node- %s' % n.name)
        
        train_mode_=graph.get_tensor_by_name('training:0')
        iter_handle =graph.get_tensor_by_name('iter_handle:0')
        keep_prob_=graph.get_tensor_by_name('keep_prob:0')
        acc_op=graph.get_tensor_by_name('acc:0')

        h,w,c = image_size
        dataset_test = ClassDatasetTFRecord_v2(w,h,c,class_num,1,batch_size,False)
        test_iter,_,_ = dataset_test(TFRecord_filepath)

        with tf.Session(config=tfconfig) as sess:
            sess.run(tf.global_variables_initializer())
            test_iter_handle=sess.run(test_iter.string_handle())
            sess.run(test_iter.initializer)
            acc_li=[]
            try:
                while True:
                    b_acc=sess.run(acc_op,feed_dict={iter_handle:test_iter_handle,keep_prob_:1,train_mode_:False})
                    acc_li.append(b_acc)
            except tf.errors.OutOfRangeError:
                pass
            test_acc=np.average(np.array(acc_li))
            print('test accuracy: '+str(test_acc))