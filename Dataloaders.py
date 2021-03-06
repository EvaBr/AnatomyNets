import torch
from torch import nn
from torch.utils import data
import glob
import nibabel as nib
from os import listdir
import numpy as np
from Slicing import get_one_hot

def get_loaders(net, dataset, n_class, in_chan, lower_in_chan, batch_size, debug):
    dub = False
    if net=="DeepMedic":
        dub = True

    data_train = DataOut(f"{dataset}/TRAIN", nb_classes=n_class, in_chan=in_chan, lower_in_chan=lower_in_chan, debug=debug, double=dub, batch_size=batch_size)
    mapa = 'VAL'
    if debug:
        mapa = 'TRAIN'
    data_val = DataOut(f"{dataset}/{mapa}", nb_classes=n_class, in_chan=in_chan, lower_in_chan=lower_in_chan, debug=debug, double=dub,  batch_size=batch_size)

    train_loader = data.DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=multi_collate, num_workers=batch_size+2, pin_memory=True)
    val_loader = data.DataLoader(data_val, batch_size=batch_size, collate_fn=multi_collate, num_workers=batch_size+2, pin_memory=True)

    return train_loader, val_loader


class DataOut(data.Dataset):
    def __init__(self, folder, nb_classes=7, in_chan=[0,1], lower_in_chan=[0,1], debug=False, double=False,  batch_size=32):
        self.folder = folder
        self.files = listdir(f"{folder}/gt/")  #assume all folders, gt, in1, in2, have same file names
        self.double = double
        self.nb_class = nb_classes
        if debug:
            self.files = self.files[:8] #:batch_size*5]
            print(self.files)
        self.in1 = in_chan 
        self.in2 = lower_in_chan
    
    def __len__(self):
        'total number of samples'
        return len(self.files)

    def __getitem__(self, item):
        'get one patch/img'
        img_name = self.files[item]
        
        inputs = []
        tmp = np.load(f"{self.folder}/in1/{img_name}")
        inputs.append(torch.tensor(tmp[self.in1,...], dtype=torch.float32))
        if self.double:
            tmp = np.load(f"{self.folder}/in2/{img_name}")
            inputs.append(torch.tensor(tmp[self.in2,...], dtype=torch.float32))
        gt = np.load(f"{self.folder}/gt/{img_name}")
        if gt.ndim<3: #this is a hardcoded 2D check that gt is one-hot encoded. when moving to 3D, change this to actual is_one_hot check!
            gt = get_one_hot(gt, self.nb_class)
        inputs.append(torch.tensor(gt, dtype=torch.float32))

        return inputs

def multi_collate(batch):
    out = []
    for i in range(len(batch[0])):
        out.append(torch.stack([item[i] for item in batch], dim=0))

    return out
