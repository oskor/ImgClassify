from .Layers import CBR2D
import tensorflow as tf

#class Resnet_V1
#class Resnet_V2

class ResnetSimple:
    def __init__(self,n,name):
        self.n_repeat=n
        self.name=name

    def __EnterBlock(self,name,input,c_out,s_0,training):
        c_in=input.shape[-1]
        x_input=input
        out=CBR2D(name+'_EnterCBR0',input,kernel=[3,3,c_in,c_out],stride=[1,s_0,s_0,1],training=training,activate=True)
        out=CBR2D(name+'_EnterCBR1',out,kernel=[3,3,c_out,c_out],stride=[1,1,1,1],training=training,activate=False)
        short_cut=CBR2D(name+'_ShortCut0',x_input,kernel=[1,1,c_in,c_out],stride=[1,s_0,s_0,1],training=training,activate=False)
        out=tf.nn.relu(out+short_cut)
        return out

    def __InterBlock(self,name,input,training):
        '''
        no change in channel num and feature size
        '''
        c_num=input.shape[-1]
        x_input=input
        out=CBR2D(name+'_InterBlock0',input,kernel=[3,3,c_num,c_num],stride=[1,1,1,1],training=training,activate=True)
        out=CBR2D(name+'_InterBlock1',out,kernel=[3,3,c_num,c_num],stride=[1,1,1,1],training=training,activate=False)
        out=tf.nn.relu(out+x_input)
        return out
    
    def __Block(self,name,input,c_out,s_0,n_repeat,training):
        out=self.__EnterBlock(name+'_n0',input,c_out,s_0,training)
        for i in range(n_repeat-1):
            out=self.__InterBlock(name+'_n'+str(i+1),out,training)
        return out

    def __call__(self,imgs,training):
        scope=self.name
        out=CBR2D(scope+'_incbr',imgs,kernel=[3,3,3,16],stride=[1,1,1,1],training=training,activate=True,padding='SAME')
        # stage0
        out=self.__Block(scope+'_stage0',out,c_out=16,s_0=1,n_repeat=self.n_repeat,training=training)
        # stage1
        out=self.__Block(scope+'_stage1',out,c_out=32,s_0=2,n_repeat=self.n_repeat,training=training)
        # stage2
        out=self.__Block(scope+'_stage2',out,c_out=64,s_0=2,n_repeat=self.n_repeat,training=training)
        # avgpool
        out=tf.nn.avg_pool(out,ksize=[1,8,8,1],strides=[1,1,1,1],padding='SAME')
        
        return out