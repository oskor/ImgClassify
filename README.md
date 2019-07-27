# ImgClassify
>自然图像分类训练挑战

|文件|说明|
|------|------|
|Config.py|所有超参数的配置|
|Cifar.py|Cifar10和Cifar100数据集加载方法，返回numpy格式的image和label
|Dataset.py|将数据集包装成迭代器形式|
|Augment.py|数据增广方法|
|Layers.py|常用的神经网络层封装
|Resnet.py|Resnet_cifar10版|
|CNNTrain.py|训练/验证函数入口|

# Config
配置好Config文件，执行CNNTrain.py，完成训练或评估。
Config分为五部分：  
* ## Stage (Train or Evalue).  
选择训练或者评估，如果是训练是否要预加载模型，如果是评估设置模型路径/数据路径
* ## Data (数据相关处理)
Precess_Func接口：  
batch_imgs=Precess_Func(batch_imgs) #在数据加载进来就调用，不管是训练还是测试
Augment_Func接口：  
batch_imgs=Augment_Func(batch_imgs) #训练阶段，Precess_Func之后调用
* ## Net (网络模型相关)
使用：
ret=Net(batch_imgs,training,keep_drop) #Net实现这个接口，或函数或class.__call__






```
class Config:
    Training=False      #Train-True / Evalue-False
    Model_Save_Dir=''
    Log_Dir=''
    Pretrained_Mode=None
    Pretrained_Mode='model1000'
    # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    variables_to_restore=None

    if Training == False:
        Evalue_Model='step_1000.pb'
        Evalue_Data_Dir='../dataset/cifar-10-python/cifar-10-batches-py/' 

    # Data
    Data_Dir='../dataset/cifar-10-python/cifar-10-batches-py/' 
    Precess_Func=None   # called on original data in both train and test stage
    # Augment_Func=Augment([1,3])    # called after Precess_Func in train stage
    Augment_Func=AugFunc

    # Net
    Image_Size=[32,32,3]  #[h,w,c]
    Class_Num=10
    Net=ResnetCifar10(9)

    # Loss
    Loss=LossType.softmax_cross_entropy_with_logits

    # Optimizer
    Base_LR=0.00001  
    LRDecay=LRPolicyType.piecewise_constant_decay
    Optimizer=OptimizerType.AdamOptimizer # adam Base_LR=0.001

    # Train routine
    Epoch=1           # The maximum number of iterations
    Test_Iter=100        # Test iter num in testing
    Test_Interval=100   # Carry out testing every 500 training iterations.
    Snapshot=500       # Snapshot intermediate results
    Batch_Size=32
```



