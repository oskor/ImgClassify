from BackBone.Resnets import ResnetSimple
from Head.ClassifyHead import ClassifyHead

class ResnetSimpleClassify:
    def __init__(self,n,class_num):
        self.bkb=ResnetSimple(n,'resnet_simple_bkb')
        self.head=ClassifyHead(class_num,'classify_head')
        
    def __call__(self,image,training,keep_prob):
        out=self.bkb(image,training)
        out=self.head(out,training,keep_prob)
        return out