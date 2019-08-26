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

       
def CBR2D(name,input,kernel,stride,training,activate,padding='SAME'):
    '''
    padding:
        SAME-out=ceil(input/s)
        VALID-out=ceil((input-kernel+1)/s)
    '''
    out=Conv2D(name+'_conv',input,kernel,stride,padding,activate=False)
    out=tf.layers.batch_normalization(out,name=name+'_bn', axis=3, training=training)
    if activate:
        out=tf.nn.relu(out)
    return out

"""
Created on Thu Aug 18 00:21:00 2019

@author: LiYang oskorliyang@163.com

layers for mobilenetv3

reference https://github.com/Bisonai/mobilenetv3-tensorflow/blob/master/layers.py
"""
def get_layer(layer_name, layer_dict, default_layer):
    if layer_name is None:
        return default_layer

    if layer_name in layer_dict.keys():
        return layer_dict.get(layer_name)
    else:
        raise NotImplementedError(f"Layer [{layer_name}] is not implemented")

def conv2d(name,input,kernel,stride,use_bias=True,padding='SAME'):
    # 注意tf.name_scope不能影响到variable，命名还是加上scope，比如:scope+'w'
    # 也可以使用tf.variable_scope，看习惯
    with tf.name_scope(name) as scope:
        w=tf.get_variable(scope+'w',
                          shape=kernel,
                          dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
        out = tf.nn.conv2d(input,w,stride,padding)
        if use_bias is True:
            n_out=kernel[-1]
            b=tf.get_variable(scope+'b',initializer=tf.constant(0.0,shape=[n_out],dtype=tf.float32))
            out=tf.nn.bias_add(out,b)       
        return out

def depthwise_conv2d(name,input,kernel,stride,use_bias=True,padding='SAME'): 
    # 注意tf.name_scope不能影响到variable，命名还是加上scope，比如:scope+'w'
    # 也可以使用tf.variable_scope，看习惯
    with tf.name_scope(name) as scope:
        w=tf.get_variable(scope+'w',
                          shape=kernel,
                          dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
        out = tf.nn.depthwise_conv2d(input,w,stride,padding)
        if use_bias is True:
            n_out=kernel[-2]*kernel[-1]
            b=tf.get_variable(scope+'b',initializer=tf.constant(0.0,shape=[n_out],dtype=tf.float32))
            out=tf.nn.bias_add(out,b)       
        return out

class cReLU(object):
    def __init__(self,name='ReLU'):
        self.name = name
    def __call__(self,input):
        return tf.nn.relu(input,name=self.name)

class cReLU6(object):
    def __init__(self,name='ReLU6'):
        self.name=name
    def __call__(self,input):
        return tf.nn.relu6(input,name=self.name)

class cSoftmax(object):
    def __init__(self,name='Softmax'):
        self.name = name
    def __call__(self,input):
        return tf.nn.softmax(input,name=self.name)

class cHSigmoid(object):
    def __init__(self,name='HSigmoid'):
        self.name = name
    def __call__(self,input):
        return tf.nn.relu6(input+3.0)/6.0

class cHSwish(object):
    def __init__(self,name='HSwish'):
        self.name = name
    def __call__(self,input):
        return input*(tf.nn.relu6(input+3.0)/6.0)

class cIdentity(object):
    def __init__(self,name='Identity'):
        self.name = name
    def __call__(self,input,is_training=True):
        return input

class cBatchNormalization(object):
    def __init__(self,name="BN"):
        self.name = name
    def __call__(self, input,is_training=True):
        return tf.layers.batch_normalization(input,training=is_training, name=self.name)

class ConvNormAct2d(object):
    def __init__(self,kernel,name:str='_',stride=[1,1,1,1],use_bias:bool=True,norm_layer:str='bn',act_layer:str='relu'):
        self.kernel = kernel
        self.name = name
        self.stride = stride
        self.use_bias = use_bias
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        _available_normalization = {
            "bn":cBatchNormalization(name=self.name+'_bn')
        }

        self.norm = get_layer(norm_layer,_available_normalization,cIdentity())

        _available_activation = {
            "relu":cReLU(name=self.name+"_ReLU"),
            "relu6":cReLU6(name=self.name+"_ReLU6"),
            "hswish":cHSwish(name=self.name+"_HSwish"),
            'hsigmoid':cHSigmoid(name=self.name+"_HSigmoid")
        }

        self.act = get_layer(act_layer,_available_activation,cIdentity())

    def __call__(self,input,is_training=True):
        x = conv2d(name=self.name+'_conv2d',input=input,kernel=self.kernel,stride=self.stride,use_bias=self.use_bias,padding='SAME')
        x = self.norm(x,is_training=is_training)
        x = self.act(x)
        return x

class cSE(object):
    def __init__(self,input_channel:int,name:str='_se',reduction:int=4):
        self.input_channel = input_channel
        self.name = name
        self.reduction = reduction

        self.s_conv = ConvNormAct2d(\
            kernel = [1,1,self.input_channel,self.input_channel/self.reduction],\
            name = self.name+'_s_conv',\
            use_bias=True,\
            norm_layer=None,\
            act_layer='relu')
        
        self.e_conv = ConvNormAct2d(\
            kernel = [1,1,self.input_channel/self.reduction, self.input_channel],\
            name = self.name+'_e_conv',\
            use_bias=True,\
            norm_layer=None,\
            act_layer='hsigmoid')

    def __call__(self,input,is_training=True):
        x = tf.reduce_mean(input, [1, 2], name=self.name+'_globe_average_pool', keep_dims=True)
        x = self.s_conv(x,is_training=is_training)
        x = self.e_conv(x,is_training=is_training)

        return input*x

class BottleNeck(object):
    def __init__(self,input_channel,expand_channel,output_channel,\
        name:str='_bneck',kernel_size:int=3,stride:int=1,use_se:bool=True,act_layer:str='relu',se_reduction:int=4):
        self.name = name
        self.exp_kernel = [1,1,input_channel,expand_channel]
        self.exp_conv = ConvNormAct2d(kernel = self.exp_kernel,\
            name=self.name+'_exp_conv',\
            use_bias=False,\
            norm_layer='bn',\
            act_layer=act_layer)

        self.dewise_kernel = [kernel_size,kernel_size,expand_channel,1]
        self.dewise_stride = [1,stride,stride,1]
        self.dewise_name   = self.name+'_dewise'   
        self.dewise_norm = cBatchNormalization(name=self.name+'_dewise_bn')

        _available_activation = {
            "relu":cReLU(name=self.name+"_dewise_ReLU"),
            "relu6":cReLU6(name=self.name+"_dewise_ReLU6"),
            "hswish":cHSwish(name=self.name+"_dewise_HSwish"),
        }

        self.dewise_act  = get_layer(act_layer,_available_activation,cIdentity())

        self.use_se = use_se
        if use_se is True:
            self.se = cSE(input_channel=expand_channel,\
                name=self.name+'_se',\
                reduction=se_reduction)
        self.red_kernel = [1,1,expand_channel,output_channel]
        self.red_conv = ConvNormAct2d(kernel = self.red_kernel,\
            name=self.name+'_red_conv',\
            use_bias=False,\
            norm_layer='bn',\
            act_layer=None)

        self.use_skip = False
        if stride==1 and input_channel == output_channel:
            self.use_skip = True

    def __call__(self,input,is_training=True):
        x = self.exp_conv(input,is_training=is_training)
        x = depthwise_conv2d(name=self.dewise_name,input=x,\
            kernel =self.dewise_kernel,stride=self.dewise_stride,use_bias=True)
        x = self.dewise_norm(x,is_training=is_training)
        x = self.dewise_act(x)
        if self.use_se is True:
            x = self.se(x,is_training=is_training)
        x = self.red_conv(x,is_training=is_training)

        if self.use_skip is True:
            x = input+x     
        return x

# """
# Created on Thu Aug 18 00:21:00 2019

# @author: LiYang oskorliyang@163.com

# layers for mobilenetv3
# reference https://github.com/Bisonai/mobilenetv3-tensorflow/blob/master/layers.py
# """
# def _make_divisible(v, divisor, min_value=None):
#     """https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor

#     return new_v

# class LayerNamespaceWrapper(tf.keras.layers.Layer):
#     """`NameWrapper` defines auxiliary layer that wraps given `layer`
#     with given `name`. This is useful for better visualization of network
#     in TensorBoard.
#     Default behavior of namespaces defined with nested `tf.keras.Sequential`
#     layers is to keep only the most high-level `tf.keras.Sequential` name.
#     """
#     def __init__(
#             self,
#             layer: tf.keras.layers.Layer,
#             name: str,
#     ):
#         super().__init__(name=name)
#         self.wrapped_layer = tf.keras.Sequential(
#             [
#                 layer,
#             ],
#             name=name,
#         )

#     def call(self, input):
#         return self.wrapped_layer(input)

# class BatchNormalization(tf.keras.layers.Layer):
#     """Searching fo MobileNetV3: All our convolutional layers
#     use batch-normalization layers with average decay of 0.99.
#     """
#     def __init__(self,momentum:float=0.99,name="BN"):
#         super().__init__(name=name)

#         self.bn = tf.keras.layers.BatchNormalization(
#             momentum=momentum,
#             name=name,
#         )

#     def call(self, input,is_training=True):
#         return self.bn(input,training=is_training)

# class Identity(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__(name="Identity")
#     def call(self,input):
#         return input

# class ReLU6(tf.keras.layers.Layer):
#     def __init__(self,name='ReLU6'):
#         super().__init__(name=name)
#         self.relu6 = tf.keras.layers.ReLU(max_value=6,name=name)
    
#     def call(self,input):
#         return self.relu6(input)

# class HSigmoid(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__(name='HSigmoid')
#         self.relu6 = ReLU6()
    
#     def call(self,input):
#         return self.relu6(input+3.0)/6.0

# class HSwish(tf.keras.layers.Layer):
#     def __init__(self,name='HSwish'):
#         super().__init__(name=name)
#         self.h_sigmoid = HSigmoid()
    
#     def call(self,input):
#         return input*self.h_sigmoid(input)

# class Squeeze(tf.keras.layers.Layer):
#     """
#     (batch,1,1,channels)->(batch,channels)
#     """
#     def __init__(self):
#         super().__init__(name='Squeeze')
#     def call(self,input):
#         x = tf.keras.backend.squeeze(input,1)
#         x = tf.keras.backend.squeeze(x,1)
#         return x

# class GlobalAveragePooling(tf.keras.layers.Layer):
#     def __init__(self,):
#         super().__init__(name="GlobalAveragePooling2D")

#     def build(self, input_shape):
#         pool_size = tuple(map(int, input_shape[1:3]))
#         self.gap = tf.keras.layers.AveragePooling2D(
#             pool_size=pool_size,
#             name=f"AvgPool{pool_size[0]}x{pool_size[1]}",
#         )

#         super().build(input_shape)

#     def call(self, input):
#         return self.gap(input)


# class CBAct(tf.keras.layers.Layer):
#     def __init__(self,filters:int, kernel_size:int=3,stride:int=1,
#         padding:int=0,norm_layer:str=None,
#         act_layer:str="relu",
#         use_bias:bool=True,
#         l2_reg:float=1e-5,name:str="CBAct"):
#         super().__init__(name=name)
#         if padding>0:
#             self.pad = tf.keras.layers.ZeroPadding2D(padding=padding,name=f"Padding{padding}X{padding}")
#         else:
#             self.pad = Identity()
#         self.conv = tf.keras.layers.Conv2D(filters=filters,
#         kernel_size=kernel_size,strides=stride,
#         name=f"Conv{kernel_size}x{kernel_size}",
#         kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
#         use_bias=use_bias)

#         _available_normalization = {
#             "bn":BatchNormalization()
#         }

#         self.norm = get_layer(norm_layer,_available_normalization,Identity())

#         _available_activation = {
#             "relu":tf.keras.layers.ReLU(name="ReLU"),
#             "relu6":ReLU6(),
#             "hswish":HSwish(),
#             'hsigmoid':HSigmoid(),
#             'softmax':tf.keras.layers.Softmax(name="Softmax")
#         }

#         self.act = get_layer(act_layer,_available_activation,Identity())
    
#     def call(self,input):
#         x = self.pad(input)
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act(x)
#         return x

# class Bneck(tf.keras.layers.Layer):
#     def __init__(
#             self,
#             out_channels: int,
#             exp_channels: int,
#             kernel_size: int,
#             stride: int,
#             use_se: bool,
#             act_layer: str,
#             l2_reg: float=1e-5,
#     ):
#         super().__init__(name="Bneck")

#         self.out_channels = out_channels
#         self.stride = stride
#         self.use_se = use_se

#         # Expand
#         self.expand = CBAct(
#             exp_channels,
#             kernel_size=1,
#             norm_layer="bn",
#             act_layer=act_layer,
#             use_bias=False,
#             l2_reg=l2_reg,
#             name="Expand"
#         )

#         # Depthwise
#         dw_padding = (kernel_size - 1) // 2
#         self.pad = tf.keras.layers.ZeroPadding2D(
#             padding=dw_padding,
#             name=f"Depthwise/Padding{dw_padding}x{dw_padding}"
#         )
#         self.depthwise = tf.keras.layers.DepthwiseConv2D(
#             kernel_size=kernel_size,
#             strides=stride,
#             name=f"Depthwise/DWConv{kernel_size}x{kernel_size}",
#             depthwise_regularizer=tf.keras.regularizers.l2(l2_reg),
#             use_bias=False
#         )
#         self.bn = BatchNormalization(name="Depthwise/BatchNormalization")
#         if self.use_se:
#             self.se = SEBottleneck(
#                 l2_reg=l2_reg,
#                 name="Depthwise/SEBottleneck"
#             )

#         _available_activation = {
#             "relu": tf.keras.layers.ReLU(name="Depthwise/ReLU"),
#             "hswish": HSwish(name="Depthwise/HardSwish"),
#         }
#         self.act = get_layer(act_layer, _available_activation, Identity())

#         # Project
#         self.project = CBAct(
#             out_channels,
#             kernel_size=1,
#             norm_layer="bn",
#             act_layer=None,
#             use_bias=False,
#             l2_reg=l2_reg,
#             name="Project"
#         )

#     def build(self, input_shape):
#         self.in_channels = int(input_shape[3])
#         super().build(input_shape)

#     def call(self, input):
#         x = self.expand(input)
#         x = self.pad(x)
#         x = self.depthwise(x)
#         x = self.bn(x)
#         if self.use_se:
#             x = self.se(x)
#         x = self.act(x)
#         x = self.project(x)

#         if self.stride == 1 and self.in_channels == self.out_channels:
#             return input + x
#         else:
#             return x

# class SEBottleneck(tf.keras.layers.Layer):
#     def __init__(
#             self,
#             reduction: int=4,
#             l2_reg: float=0.01,
#             name: str="SEBottleneck",
#     ):
#         super().__init__(name=name)

#         self.reduction = reduction
#         self.l2_reg = l2_reg

#     def build(self, input_shape):
#         input_channels = int(input_shape[3])
#         self.gap = GlobalAveragePooling()
#         self.conv1 = CBAct(
#             input_channels // self.reduction,
#             kernel_size=1,
#             norm_layer=None,
#             act_layer="relu",
#             use_bias=False,
#             l2_reg=self.l2_reg,
#             name="Squeeze"
#         )
#         self.conv2 = CBAct(
#             input_channels,
#             kernel_size=1,
#             norm_layer=None,
#             act_layer="hsigmoid",
#             use_bias=False,
#             l2_reg=self.l2_reg,
#             name="Excite"
#         )

#         super().build(input_shape)

#     def call(self, input):
#         x = self.gap(input)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return input * x