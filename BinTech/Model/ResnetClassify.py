from ..BackBone.Resnets import ResnetSimple
from ..BackBone.resnet_v0 import resnet_v2_a3
from ..Head.ClassifyHead import ClassifyHead

class ResnetSimpleClassify:
    def __init__(self,n,class_num):
        self.bkb=ResnetSimple(n,'resnet_simple_bkb')
        self.head=ClassifyHead(class_num,'classify_head')
        
    def __call__(self,image,training,keep_prob):
        out=self.bkb(image,training)
        out=self.head(out,training,keep_prob)
        return out

class resnet_v2_a3_mark:
    def __init__(self,class_num):
        self.class_num = class_num
    def __call__(self,image,training,keep_prob):
        out = resnet_v2_a3(image,self.class_num,training,keep_prob)
        return out