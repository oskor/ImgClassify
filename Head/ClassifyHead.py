from Layers import *
import tensorflow as tf

class ClassifyHead:
    def __init__(self,class_num,name):
        self.class_num=class_num
        self.name=name
    
    def __call__(self,image,training,keep_prob):
        flatten=tf.layers.flatten(image)
        out=FC(self.name,flatten,self.class_num,activate=False)
        return out