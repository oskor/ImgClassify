import numpy as np

class Augment:
    def __init__(self,rotate_range):
        self.rotate=[]
    
    def __call__(self,batch_imgs):
        return batch_imgs

def AugFunc(b_imgs):
    # b_imgs[b,h,w,c]
    # extent 4 edge in h/w
    expand_imgs=np.pad(b_imgs,((0,0),(4,4),(4,4),(0,0)),mode='constant')
    new_imgs=np.zeros_like(b_imgs)
    b=b_imgs.shape[0]
    w_rand=np.random.randint(0,8,b)
    h_rand=np.random.randint(0,8,b)
    f_rand=np.random.randint(0,2,b)
    for i in range(b):       
        temp=expand_imgs[i,h_rand[i]:h_rand[i]+32,w_rand[i]:w_rand[i]+32,:]
        if f_rand[i] == 1:
            new_imgs[i,:]=np.flip(temp,1)
        else:
            new_imgs[i,:]=temp
    return new_imgs