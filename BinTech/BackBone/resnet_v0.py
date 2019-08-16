import tensorflow as tf

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x ,W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def conv2d_layer(input, W, name):
    with tf.name_scope('conv1'):
        W_conv = weight_variable(W)
        b_conv = bias_variable(W[3])
        net = tf.nn.relu(conv2d(input, W_conv) + b_conv)
    return net

def res_block_v0(inputs,
              channel_out,
              kernel_size=[3, 3],
              is_training=True,
              scope=None):

    """

    :param inputs:
    :param channel_out:
    :param kernel_size:
    :param is_training:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'res_block_v0', [inputs]) as sc:
        shortcut = inputs

        residual = inputs

        residual = tf.layers.conv2d(
            residual, channel_out, kernel_size, strides=[1, 1], padding='same', name='conv1')
        residual = tf.layers.batch_normalization(residual, training=is_training, name='BN1')
        residual = tf.nn.relu(residual, name='ReLU1')

        residual = tf.layers.conv2d(
            residual, channel_out, kernel_size, strides=[1, 1], padding='same', name='conv2')
        residual = tf.layers.batch_normalization(residual, training=is_training, name='BN2')

        net = residual + shortcut
        net = tf.nn.relu(net, name='ReLU2')

        return net


def res_block_v1(inputs,
              channel_out,
              kernel_size=[3, 3],
              stride=1,
              is_training=True,
              scope=None):
    """
    simple residual block repeat convolution with ReLU activation

    :param inputs: input tensor
    :param channel_out: output channel number
    :param kernel_size: 2D convolution kernel, default as [3, 3]
    :param stride:      stride, default as 1
    :param is_training: bool index for batch normalization if is training mode
    :param scope:       variable scope of the block
    :return:            output tensor
    """

    with variable_scope.variable_scope(scope, 'res_block_v1', [inputs]) as sc:
        channel_in = inputs.shape[len(inputs.shape) - 1]

        residual = inputs

        if channel_out == channel_in:
            shortcut = inputs
        else:
            shortcut = tf.layers.conv2d(
                inputs, channel_out, [1, 1], strides=[stride, stride], padding='same', name='shortcut')

        #residual = inputs
        residual = tf.layers.conv2d(
            residual, channel_out, kernel_size, strides=[stride, stride], padding='same', name='conv1')
        residual = tf.layers.batch_normalization(residual, training=is_training, name='BN1')
        residual = tf.nn.relu(residual, name='ReLU1')

        residual = tf.layers.conv2d(
            residual, channel_out, kernel_size, strides=[1, 1], padding='same', name='conv2')
        residual = tf.layers.batch_normalization(residual, training=is_training, name='BN2')

        net = residual + shortcut
        net = tf.nn.relu(net, name='ReLU_out')

        return net

def res_block_v2(inputs,
              channel_out,
              channel_mid,
              kernel_size=[3, 3],
              stride=1,
              is_training=True,
              scope=None):
    """
    simple residual block repeat convolution with ReLU activation

    :param inputs: input tensor
    :param channel_out: output channel number
    :param channel_mid: middle channel number
    :param kernel_size: 2D convolution kernel, default as [3, 3]
    :param stride:      stride, default as 1
    :param is_training: bool index for batch normalization if is training mode
    :param scope:       variable scope of the block
    :return:            output tensor
    """

    with variable_scope.variable_scope(scope, 'res_block_v2', [inputs]) as sc:
        channel_in = inputs.shape[len(inputs.shape) - 1]

        residual = inputs

        if channel_out == channel_in:
            shortcut = inputs
        else:
            shortcut = tf.layers.conv2d(
                inputs, channel_out, [1, 1], strides=[stride, stride], padding='same', name='shortcut')

        #residual = inputs
        residual = tf.layers.conv2d(
            residual, channel_mid, [1, 1], strides=[stride, stride], padding='same', name='conv1')
        residual = tf.layers.batch_normalization(residual, training=is_training, name='BN1')
        residual = tf.nn.relu(residual, name='ReLU1')

        residual = tf.layers.conv2d(
            residual, channel_mid, kernel_size, strides=[1, 1], padding='same', name='conv2')
        residual = tf.layers.batch_normalization(residual, training=is_training, name='BN2')
        residual = tf.nn.relu(residual, name='ReLU2')

        residual = tf.layers.conv2d(
            residual, channel_out, [1, 1], strides=[1, 1], padding='same', name='conv3')
        residual = tf.layers.batch_normalization(residual, training=is_training, name='BN3')

        net = residual + shortcut
        net = tf.nn.relu(net, name='ReLU_out')

        return net

def resnet_v0(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v0'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v0', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 64, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name='BN1')
        net = tf.nn.relu(net, name='ReLU1')

        net = res_block_v0(net, 64, [3, 3], is_training=is_training, scope='res_block2')
        net = res_block_v0(net, 64, [3, 3], is_training=is_training, scope='res_block3')

        net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool4')
        net = tf.layers.conv2d(
            net, 128, [1, 1], strides=[1, 1], padding='same', name='pool4_out')

        net = res_block_v0(net, 128, [3, 3], is_training=is_training, scope='res_block5')
        net = res_block_v0(net, 128, [3, 3], is_training=is_training, scope='res_block6')

        net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool7')
        net = tf.layers.conv2d(
            net, 256, [1, 1], strides=[1, 1], padding='same', name='pool7_out')

        net = res_block_v0(net, 256, [3, 3], is_training=is_training, scope='res_block8')
        net = res_block_v0(net, 256, [3, 3], is_training=is_training, scope='res_block9')

        net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool10')

        #net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        shape = net.shape
        net = tf.layers.conv2d(
           net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v0_a(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v0'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v0_a', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 16, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name='BN1')
        net = tf.nn.relu(net, name='ReLU1')

        net = res_block_v0(net, 16, [3, 3], is_training=is_training, scope='res_block2')
        net = res_block_v0(net, 16, [3, 3], is_training=is_training, scope='res_block3')
        net = res_block_v0(net, 16, [3, 3], is_training=is_training, scope='res_block4')

        net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool4')
        net = tf.layers.conv2d(
            net, 32, [1, 1], strides=[1, 1], padding='same', name='pool4_out')

        net = res_block_v0(net, 32, [3, 3], is_training=is_training, scope='res_block5')
        net = res_block_v0(net, 32, [3, 3], is_training=is_training, scope='res_block6')
        net = res_block_v0(net, 32, [3, 3], is_training=is_training, scope='res_block7')

        net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool7')
        net = tf.layers.conv2d(
            net, 64, [1, 1], strides=[1, 1], padding='same', name='pool7_out')

        net = res_block_v0(net, 64, [3, 3], is_training=is_training, scope='res_block8')
        net = res_block_v0(net, 64, [3, 3], is_training=is_training, scope='res_block9')
        net = res_block_v0(net, 64, [3, 3], is_training=is_training, scope='res_block10')

        net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool10')

        #net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        shape = net.shape
        net = tf.layers.conv2d(
           net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v1_a(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v1_a'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v1_a', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 16, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block3')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block4')
        net = res_block_v1(net, 32, [3, 3], stride=2, is_training=is_training, scope='res_block5')
        net = res_block_v1(net, 32, [3, 3], stride=1, is_training=is_training, scope='res_block6')
        net = res_block_v1(net, 32, [3, 3], stride=1, is_training=is_training, scope='res_block7')
        net = res_block_v1(net, 64, [3, 3], stride=2, is_training=is_training, scope='res_block8')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block9')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block10')

        #net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool')

        net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        #shape = net.shape
        #net = tf.layers.conv2d(
        #   net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')

        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v1_a1(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v1_a1'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v1_a1', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 16, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block3')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block4')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block5')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block6')
        net = res_block_v1(net, 16, [3, 3], stride=1, is_training=is_training, scope='res_block7')
        net = res_block_v1(net, 32, [3, 3], stride=2, is_training=is_training, scope='res_block8')
        net = res_block_v1(net, 32, [3, 3], stride=1, is_training=is_training, scope='res_block9')
        net = res_block_v1(net, 32, [3, 3], stride=1, is_training=is_training, scope='res_block10')
        net = res_block_v1(net, 32, [3, 3], stride=1, is_training=is_training, scope='res_block11')
        net = res_block_v1(net, 32, [3, 3], stride=1, is_training=is_training, scope='res_block12')
        net = res_block_v1(net, 32, [3, 3], stride=1, is_training=is_training, scope='res_block13')
        net = res_block_v1(net, 64, [3, 3], stride=2, is_training=is_training, scope='res_block14')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block15')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block16')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block17')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block18')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block19')

        net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool')

        #net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        shape = net.shape
        net = tf.layers.conv2d(
           net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v1_a0(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v1_a0'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v1_a0', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 64, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block3')

        net = res_block_v1(net, 128, [3, 3], stride=2, is_training=is_training, scope='res_block4')
        net = res_block_v1(net, 128, [3, 3], stride=1, is_training=is_training, scope='res_block5')

        net = res_block_v1(net, 256, [3, 3], stride=2, is_training=is_training, scope='res_block6')
        net = res_block_v1(net, 256, [3, 3], stride=1, is_training=is_training, scope='res_block7')

        net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool8')

        #net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        shape = net.shape
        net = tf.layers.conv2d(
           net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v1_a2(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v1_a2'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v1_a2', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 64, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v1(net, 64, [3, 3], stride=1, is_training=is_training, scope='res_block3')

        net = res_block_v1(net, 128, [3, 3], stride=2, is_training=is_training, scope='res_block4')
        net = res_block_v1(net, 128, [3, 3], stride=1, is_training=is_training, scope='res_block5')

        net = res_block_v1(net, 256, [3, 3], stride=2, is_training=is_training, scope='res_block6')
        net = res_block_v1(net, 256, [3, 3], stride=1, is_training=is_training, scope='res_block7')

        #net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool8')

        net = tf.layers.conv2d(net, 1024, [1, 1], strides=[1, 1], padding='same', name='feature_maps')
        net = tf.layers.batch_normalization(net, training=is_training, name='feature_maps_BN')
        net = tf.nn.relu(net, name='feature_maps_ReLU')

        net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        #shape = net.shape
        #net = tf.layers.conv2d(
        #   net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v2_a(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v2_a'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v2_a', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 32, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v2(net, 32, 16, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v2(net, 32, 16, [3, 3], stride=1, is_training=is_training, scope='res_block3')
        net = res_block_v2(net, 32, 16, [3, 3], stride=1, is_training=is_training, scope='res_block4')
        net = res_block_v2(net, 64, 32, [3, 3], stride=2, is_training=is_training, scope='res_block5')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block6')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block7')
        net = res_block_v2(net, 128, 64, [3, 3], stride=2, is_training=is_training, scope='res_block8')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block9')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block10')

        #net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool')

        net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        #shape = net.shape
        #net = tf.layers.conv2d(
        #   net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')

        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v2_a0(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v2_a0'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v2_a0', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 64, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block3')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block4')
        net = res_block_v2(net, 128, 64, [3, 3], stride=2, is_training=is_training, scope='res_block5')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block6')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block7')
        net = res_block_v2(net, 256, 128, [3, 3], stride=2, is_training=is_training, scope='res_block8')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block9')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block10')

        #net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool')

        net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        #shape = net.shape
        #net = tf.layers.conv2d(
        #   net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')

        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v2_a1(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v2_a1'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v2_a1', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 64, [5, 5], strides=[1, 1], padding='valid', name='conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name='BN1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block3')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block4')
        net = res_block_v2(net, 128, 64, [3, 3], stride=2, is_training=is_training, scope='res_block5')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block6')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block7')
        net = res_block_v2(net, 256, 128, [3, 3], stride=2, is_training=is_training, scope='res_block8')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block9')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block10')

        #net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool')

        net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        #shape = net.shape
        #net = tf.layers.conv2d(
        #   net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')

        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v2_a2(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v2_a2'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v2_a2', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 64, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block3')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block4')
        net = res_block_v2(net, 128, 64, [3, 3], stride=2, is_training=is_training, scope='res_block5')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block6')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block7')
        net = res_block_v2(net, 256, 128, [3, 3], stride=2, is_training=is_training, scope='res_block8')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block9')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block10')
        net = res_block_v2(net, 512, 256, [3, 3], stride=2, is_training=is_training, scope='res_block11')
        net = res_block_v2(net, 512, 256, [3, 3], stride=1, is_training=is_training, scope='res_block12')
        net = res_block_v2(net, 512, 256, [3, 3], stride=1, is_training=is_training, scope='res_block13')

        #net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool')

        net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        #shape = net.shape
        #net = tf.layers.conv2d(
        #   net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')

        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net

def resnet_v2_a3(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='resnet_v2_a3'):

    """

    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :return:
    """

    with variable_scope.variable_scope(scope, 'resnet_v2_a3', [inputs]) as sc:
        net = tf.layers.conv2d(
            inputs, 64, [3, 3], strides=[1, 1], padding='same', name='conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name='BN1')
        net = tf.nn.relu(net, name='ReLU1')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block2')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block3')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block4')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block5')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block6')
        net = res_block_v2(net, 64, 32, [3, 3], stride=1, is_training=is_training, scope='res_block7')
        net = res_block_v2(net, 128, 64, [3, 3], stride=2, is_training=is_training, scope='res_block8')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block9')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block10')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block11')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block12')
        net = res_block_v2(net, 128, 64, [3, 3], stride=1, is_training=is_training, scope='res_block13')
        net = res_block_v2(net, 256, 128, [3, 3], stride=2, is_training=is_training, scope='res_block14')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block15')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block16')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block17')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block18')
        net = res_block_v2(net, 256, 128, [3, 3], stride=1, is_training=is_training, scope='res_block19')

        #net = tf.layers.average_pooling2d(net, [2, 2], strides=[2, 2], padding='same', name='pool')

        net = tf.reduce_mean(net, [1, 2], name='globe_average_pool', keep_dims=True)

        #shape = net.shape
        #net = tf.layers.conv2d(
        #   net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')

        net = tf.nn.relu(net, name='fc_ReLU')
        net = tf.layers.dropout(net, rate=1.0 - dropout_keep_prob, training=is_training, name='dropout')

        #net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #shape = net.shape
        #net = tf.layers.conv2d(
        #    net, 1024, [shape[1], shape[2]], strides=[1, 1], padding='valid', name='fc_feature')
        #net = tf.nn.relu(net, name='fc_ReLU')

        net = tf.layers.conv2d(
            net, num_classes, [1, 1], strides=[1, 1], padding='same', name='fc')

        if spatial_squeeze:
            net = array_ops.squeeze(net, [1, 2], name='fc/squeezed')
        return net