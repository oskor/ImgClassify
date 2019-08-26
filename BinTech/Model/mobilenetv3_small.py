import tensorflow as tf
from ..BackBone.Layers import ConvNormAct2d,BottleNeck
from ..Head.ClassifyHead import ClassifyHead_mobilev3 as LastStage

class MobileNetV3(object):
    def __init__(self,num_class,name:str='mobilenetv3_small',image_channel:int=3):
        self.num_class = num_class
        self.name = name
        self.first_layer_output_channel = 16
        self.first_layer_stride = 1
        self.first_layer = ConvNormAct2d(kernel=[3,3,image_channel, self.first_layer_output_channel],name='first_layer',\
            stride=[1,self.first_layer_stride,self.first_layer_stride,1],norm_layer='bn',act_layer='hswish')
        
        # Bottleneck layers
        self.bneck_settings = [
             # k   exp   out  SE      NL         s
             [ 3,  16,   16,  True,   "relu",    1 ],
             [ 3,  72,   24,  False,  "relu",    2 ],
             [ 3,  88,   24,  False,  "relu",    1 ],
             [ 5,  96,   40,  True,   "hswish",  2 ],
             [ 5,  240,  40,  True,   "hswish",  1 ],
             [ 5,  240,  40,  True,   "hswish",  1 ],
             [ 5,  120,  48,  True,   "hswish",  1 ],
             [ 5,  144,  48,  True,   "hswish",  1 ],
             [ 5,  288,  96,  True,   "hswish",  2 ],
             [ 5,  576,  96,  True,   "hswish",  1 ],
             [ 5,  576,  96,  True,   "hswish",  1 ],
         ] 

        self.bneck = []

        last_input_channel = self.first_layer_output_channel

        for idx, (k, exp, out, SE, NL, s) in enumerate(self.bneck_settings):
            bneck_i = BottleNeck(input_channel = last_input_channel,\
                expand_channel = exp,\
                output_channel = out,\
                name = 'bneck_'+str(idx),\
                kernel_size=k,stride=s,use_se=SE,act_layer=NL)
            last_input_channel = out
            self.bneck.append(bneck_i)

        penultimate_channel = 256
        last_channel = 512
        self.last_layer = LastStage(last_input_channel,penultimate_channel,last_channel,num_class,'last_layer')

    def __call__(self,input,training,keep_drop):
        input = self.first_layer(input,is_training=training)
        for bneck in self.bneck:
            input = bneck(input,is_training=training)
        input =self.last_layer(input,training=training,keep_drop=keep_drop)

        return input



# from ..BackBone.Layers import CBAct
# from ..BackBone.Layers import Bneck
# from ..Head.ClassifyHead import ClassifyHead_mobilev3 as LastStage

# from ..BackBone.Layers import _make_divisible
# from ..BackBone.Layers import LayerNamespaceWrapper


# class MobileNetV3(object):
#     def __init__(
#             self,
#             num_classes: int=10,
#             width_multiplier: float=1.0,
#             name: str="MobileNetV3_Small",
#             divisible_by: int=8,
#             l2_reg: float=1e-5,
#     ):
#         #super().__init__(name=name)

#         # First layer
#         self.first_layer = CBAct(
#             16,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             norm_layer="bn",
#             act_layer="hswish",
#             use_bias=False,
#             l2_reg=l2_reg,
#             name="FirstLayer",
#         )

#         # Bottleneck layers
#         self.bneck_settings = [
#             # k   exp   out  SE      NL         s
#             [ 3,  16,   16,  True,   "relu",    1 ],
#             #[ 3,  72,   24,  False,  "relu",    1 ],
#             [ 3,  88,   24,  False,  "relu",    1 ],
#             [ 5,  96,   40,  True,   "hswish",  2 ],
#             #[ 5,  240,  40,  True,   "hswish",  1 ],
#             [ 5,  240,  40,  True,   "hswish",  1 ],
#            # [ 5,  120,  48,  True,   "hswish",  1 ],
#             [ 5,  144,  48,  True,   "hswish",  1 ],
#             [ 5,  288,  96,  True,   "hswish",  2 ],
#             [ 5,  576,  96,  True,   "hswish",  1 ],
#             #[ 5,  576,  96,  True,   "hswish",  1 ],
#         ]

#         self.bneck = tf.keras.Sequential(name="Bneck")
#         for idx, (k, exp, out, SE, NL, s) in enumerate(self.bneck_settings):
#             out_channels = _make_divisible(out * width_multiplier, divisible_by)
#             exp_channels = _make_divisible(exp * width_multiplier, divisible_by)

#             self.bneck.add(
#                 LayerNamespaceWrapper(
#                     Bneck(
#                         out_channels=out_channels,
#                         exp_channels=exp_channels,
#                         kernel_size=k,
#                         stride=s,
#                         use_se=SE,
#                         act_layer=NL,
#                     ),
#                     name=f"Bneck{idx}")
#             )

#         # Last stage
#         penultimate_channels = 128#_make_divisible(576 * width_multiplier, divisible_by)
#         last_channels = 256#_make_divisible(1280 * width_multiplier, divisible_by)

#         self.last_stage = LastStage(
#             penultimate_channels,
#             last_channels,
#             num_classes,
#             0.5,
#             l2_reg=l2_reg,
#             name=f"LastStage"
#         )

#     def __call__(self, input):
#         x = self.first_layer(input)
#         x = self.bneck(x)
#         x = self.last_stage(x)
#         return x