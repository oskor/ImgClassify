# ImgClassify
>自然图像分类训练挑战

|文件|说明|扩充|
|------|------|------|
|Config.py|所有超参数的配置|
|Cifar.py|Cifar10和Cifar100数据集加载方法
|Dataset.py|将数据集包装成迭代器形式|李阳完成替换|
|Augment.py|数据增广方法|任远完成替换
|Layers.py|常用的神经网络层封装
|Resnet.py|Resnet_cifar10版|
|CNNTrain.py|训练/验证函数入口|震寰加入tensorboard部分|

# Config
配置好Config文件，就可以执行CNNTrain.py完成训练或评估。

Config分为五部分(**下面的接口可以是函数，也可以是class.__call\__**)：  
* ## Stage (Train or Evalue).  
选择训练或者评估，如果是训练是否要预加载模型，如果是评估设置模型路径/数据路径
* ## Data (数据相关处理)
Precess_Func接口：  
batch_imgs=Precess_Func(batch_imgs) #在数据加载进来就调用，不管是训练还是测试  
Augment_Func接口：  
batch_imgs=Augment_Func(batch_imgs) #训练阶段，Precess_Func之后调用  
* ## Net (网络模型相关)
Net接口：  
ret=Net(batch_imgs,training,keep_drop)
* ## Loss  
可使用的loss类型参照ConfigType.py  
* ## Optimizer
可使用的Optimizer参照ConfigType.py
