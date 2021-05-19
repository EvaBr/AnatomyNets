import numpy as np
import torch

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