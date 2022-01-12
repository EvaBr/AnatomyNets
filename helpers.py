import numpy as np
import torch
import re

def getpid(filnamn):
    found = re.findall(r'500[0-9]+', filnamn) + [None]
    return found[0]

def get_one_hot(gt, nb_class):
    gt = gt.astype('int')
    classes = np.eye(nb_class)
    one_hot = classes[gt]
    s = np.arange(one_hot.ndim)
    return np.transpose(one_hot, (s[-1],*s[:-1]))

def flatten_one_hot(one_hot): #OBS no batch !
    return np.argmax(one_hot, axis=0)

def CenterCropTensor(tgt, x):
    xs2, xs3 = x.shape[-2], x.shape[-1]
    tg2, tg3 = tgt.shape[-2], tgt.shape[-1]
    diffY = abs(xs2 - tg2)//2
    diffX = abs(xs3 - tg3)//2
    ostanek = abs(xs2-tg2)%2
    
    if xs2>tg2: 
        x = x[..., diffX:xs2-diffX-ostanek, diffY:xs3-diffY-ostanek]
    else: 
        tgt = tgt[..., diffX:tg2-diffX-ostanek, diffY:tg3-diffY-ostanek] 
    return tgt, x

def CenterCropTensor3d(tgt, x):
    xs1, xs2, xs3 = x.shape[-1], x.shape[-2], x.shape[-3]
    tg1, tg2, tg3 = tgt.shape[-1], tgt.shape[-2], tgt.shape[-3]
    diffZ = abs(xs1 - tg1)//2
    diffY = abs(xs2 - tg2)//2
    diffX = abs(xs3 - tg3)//2
    ostanekz = abs(xs1-tg1)%2
    ostaneky = abs(xs2-tg2)%2
    ostanekx = abs(xs3-tg3)%2
    
    if xs2>tg2:
        #we assume that either tgt or x is larger in ALL dimensions (cant be mixed)
        x = x[..., diffX:xs3-diffX-ostanekx, diffY:xs2-diffY-ostaneky,  diffZ:xs1-diffX-ostanekz]
    else: 
        tgt = tgt[..., diffX:tg3-diffX-ostanekx, diffY:tg2-diffY-ostaneky, diffZ:tg1-diffZ-ostanekz]
        
    return tgt, x

