import tensorflow as tf
from ConfigType import *


def get_optimizer(config,lr):
    cur_optimizer=config.Optimizer
    if cur_optimizer  == OptimizerType.GradientDescentOptimizer:
        opt=tf.train.GradientDescentOptimizer(lr)
    elif cur_optimizer == OptimizerType.AdamOptimizer:
        opt=tf.train.AdamOptimizer(lr,cur_optimizer.beta1,cur_optimizer.beta2,cur_optimizer.epsilon)
    else:
        print('not a avaliable optimizer in Config !')
        opt=None
    
    return opt

def get_LRPolicy(config,global_step):
    cur_LRDecay=config.LRDecay
    if  cur_LRDecay == LRPolicyType.constant:
        decay=config.Base_LR
    elif cur_LRDecay == LRPolicyType.piecewise_constant_decay:
        values=cur_LRDecay.values.copy()
        values.insert(0,config.Base_LR)
        decay=tf.train.piecewise_constant(global_step,cur_LRDecay.boundaries,values)
    elif cur_LRDecay == LRPolicyType.exponential_decay:
        decay=tf.train.exponential_decay(config.Base_LR,global_step,cur_LRDecay.decay_steps,cur_LRDecay.decay_rate,cur_LRDecay.staircase)
    elif cur_LRDecay == LRPolicyType.natural_exp_decay:
        decay=tf.train.natural_exp_decay(config.Base_LR,global_step,cur_LRDecay.decay_steps,cur_LRDecay.decay_rate,cur_LRDecay.staircase)
    elif cur_LRDecay == LRPolicyType.cosine_decay:
        decay=tf.train.cosine_decay(config.Base_LR,global_step,cur_LRDecay.decay_steps,cur_LRDecay.alpha)
    else:
        print('not a valiable LR Decay method in Config')
        decay=None
    return decay

def get_loss(config,labels, logits):
    with tf.name_scope('loss'):
        cur_loss=config.Loss
        if cur_loss == LossType.softmax_cross_entropy_with_logits:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(cross_entropy)
        else:
            print('not a valiable LR Decay method in Config')
            loss=None
        return loss


if __name__=='__main__':
    opt=get_optimizer(Config,0.3)
    print(opt)