# ImgClassify
>自然图像分类训练挑战

|文件|说明|扩充建议|
|------|------|------|
|Cifar.py|Cifar10和Cifar100数据集加载方法，返回numpy格式的image和label|对于其它数据集的python加载函数，比如：ImageNet100.py|
|Augmentation.py|数据扩增方法，在Dataset中会用到|需要增加扩充方法
|Dataset.py|将Numpy数据包装成迭代器，以方便训练/测试使用|其它数据迭代器，比如: tf.data.Dataset，急需补充TFRecord部分|
|Layers.py|常用的神经网络层封装|其它常用网络层，比如BN/CBR等|
|Lenet.py|Lenet5和Lenet_Config|其它分类网络及独有配置，比如：VGG&VGG_Config/ResNet&ResNet_Config等|
|CNNTrain.py|训练函数入口|其它训练策略，比如OHEM等

# 后续扩充方向
- 上表扩充建议中提到的内容
- 缺乏运行设备控制机制。如果有NVIDIA GPU，默认选择GPU:0运行，没有就CPU运行。且不支持CPU多线程。



