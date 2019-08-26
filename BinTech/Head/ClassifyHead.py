from ..BackBone.Layers import FC,ConvNormAct2d
import tensorflow as tf
from tensorflow.python.ops import array_ops

class ClassifyHead:
    def __init__(self,class_num,name):
        self.class_num=class_num
        self.name=name
    
    def __call__(self,image,training,keep_prob):
        flatten=tf.layers.flatten(image)
        out=FC(self.name,flatten,self.class_num,activate=False)
        return out


class ClassifyHead_mobilev3(ClassifyHead):
    def __init__(self,input_channels,penultimate_channels,last_channels,num_classes,name='LastStage'):
        super().__init__(num_classes,name)
        self.penultimate_channels = penultimate_channels
        self.last_channels = last_channels

        self.conv1=ConvNormAct2d(kernel=[1,1,input_channels,penultimate_channels],name=name+'_penultimate_conv',use_bias=False,\
            norm_layer='bn',act_layer='hswish')
        self.conv2=ConvNormAct2d(kernel=[1,1,penultimate_channels,last_channels],name=name+'_last_conv',use_bias=True,\
            norm_layer=None,act_layer='hswish')
        self.conv3=ConvNormAct2d(kernel=[1,1,last_channels,num_classes],name=name+'_class_conv',use_bias=True,\
            norm_layer=None,act_layer=None)
    
    def __call__(self,input,training,keep_drop=0.5):
        x = self.conv1(input,is_training=training)
        x = tf.reduce_mean(x, [1, 2], name=self.name+'_globe_average_pool', keep_dims=True)
        x = self.conv2(x,is_training=training)
        x = tf.layers.dropout(x, rate=1.0 - keep_drop, training=training, name=self.name+'/dropout')
        x = self.conv3(x,is_training=training)

        return array_ops.squeeze(x, [1, 2], name=self.name+'/fc/squeezed')


# class ClassifyHead_mobilev3(tf.keras.layers.Layer):
#     def __init__(self,
#             penultimate_channels: int,
#             last_channels: int,
#             num_classes: int,
#             drop_out_rate=0.5,
#             l2_reg: float=1e-5,
#             name:str=f"LastStage"):
#         super().__init__(name=name)

#         self.conv1 = CBAct(
#             penultimate_channels,
#             kernel_size=1,
#             stride=1,
#             norm_layer="bn",
#             act_layer="hswish",
#             use_bias=False,
#             l2_reg=l2_reg,
#         )
#         self.gap = GlobalAveragePooling()
#         self.conv2 = CBAct(
#             last_channels,
#             kernel_size=1,
#             norm_layer=None,
#             act_layer="hswish",
#             l2_reg=l2_reg,
#         )
#         self.dropout = tf.keras.layers.Dropout(
#             rate=drop_out_rate,
#             name=f"Dropout",
#         )
#         self.conv3 = CBAct(
#             num_classes,
#             kernel_size=1,
#             norm_layer=None,
#             act_layer="softmax",
#             l2_reg=l2_reg,
#         )
#         self.squeeze = Squeeze()

#     def call(self, input):
#         x = self.conv1(input)
#         x = self.gap(x)
#         x = self.conv2(x)
#         x = self.dropout(x)
#         x = self.conv3(x)
#         x = self.squeeze(x)
#         return x
    

