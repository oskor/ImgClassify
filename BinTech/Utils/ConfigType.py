import tensorflow

class OptimizerType:
    class GradientDescentOptimizer:
        # there is no more parameter for SGD，even momentum 
        _=1
    class AdamOptimizer:
        beta1=0.9
        beta2=0.999
        epsilon=1e-8

class LRPolicyType:
    class constant:
        _=1
    class piecewise_constant_decay:        
        boundaries=[15000,30000,75000]
        values=[0.003,0.0009,0.0005]
        
    class exponential_decay:
        # lr=Base_LR * decay_rate ^ (cur_step / decay_steps)
        decay_steps=3000
        decay_rate=0.5
        staircase=True  
        
    class natural_exp_decay:
        # lr=Base_LR * Exp(-decay_rate*[cur_step / decay_steps])
        decay_steps=100
        decay_rate=0.5
        staircase=True  
    
    class cosine_decay:
        # Using cos func to scale lr into[alpha,Base_LR]
        decay_steps=100
        alpha=0.000001 # mininum lr
    
class LossType:
    class softmax_cross_entropy_with_logits:
        # # there is no more parameter for softmax_cross_entropy_with_logits
        _=1