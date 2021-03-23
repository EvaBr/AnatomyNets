import numpy as np
import nibabel as nib 
from glob import glob
import os
import random
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
####################################
def get_one_hot(gt, nb_class):
    gt = gt.astype('int')
    classes = np.eye(nb_class)
    one_hot = classes[gt]
    s = np.arange(one_hot.ndim)
    return np.transpose(one_hot, (s[-1],*s[:-1]))

def flatten_one_hot(one_hot):
    return np.argmax(one_hot, axis=0)

def check2Dcuts(pid, inp2=False):
    in1 = np.load(f"./POEM/in1/subj{pid}.npy")
    if inp2:
        in2 = np.load(f"./POEM/in2/subj{pid}.npy")
    gt = flatten_one_hot(np.load(f"./POEM/gt/subj{pid}.npy"))

    plt.figure(figsize=(20,7))
    plt.subplot(1,3,1)
    plt.imshow(gt)
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(in1[0,...]))
    if inp2:
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(in2[0,...]))
    plt.show()
   
#%%
check2Dcuts("500077_30_0", True)
check2Dcuts("500077_30_1", True)
check2Dcuts("500297_30_0", True)
check2Dcuts("500297_30_1", True)
####################################


outpath = "POEM/TRAIN"

add_dts = False
make_subsampled = True
nb_class = 7
patch_size = 50

#POEM SLICING
gt_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segment_all/converted/CroppedSegmNew*")
wat_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/cropped*_wat*")
fat_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/cropped*_fat*")
dtx_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/distmaps/*x.nii")
dty_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/distmaps/*y.nii")
mask_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/cropped*_mask.nii")

gt_paths.sort()
wat_paths.sort()
fat_paths.sort()
dtx_paths.sort()
dty_paths.sort()
mask_paths.sort()

#%%
if not os.path.exists(outpath):
    os.mkdir(outpath)
for i in ['gt','in1','in2']:
    if not os.path.exists(f"{outpath}{i}"):
        os.mkdir(f"{outpath}{i}")

#%%
def cut():
    patch = patch_size//2
    for w,f,g,dx,dy,m in zip(wat_paths, fat_paths, gt_paths, dtx_paths, dty_paths, mask_paths):
        PID = re.findall(r"500[0-9]+", w)[0]
        print(f"Slicing nr {PID}...")
        wat = nib.load(w).get_fdata()
        fat = nib.load(f).get_fdata()
        gt = nib.load(g).get_fdata()
        x = nib.load(dx).get_fdata()
        y = nib.load(dy).get_fdata()
        maska = nib.load(m).get_fdata()

        gt = get_one_hot(gt, nb_class) #new size C x H x W x D


        for slajs in tqdm(range(wat.shape[1])):
            inx = wat.shape[0]//patch_size
            iny = wat.shape[2]//patch_size
            to_cut = 2 #max(4, inx*iny)

            wat_tmp = np.pad(np.squeeze(wat[:, slajs, :]),(patch+16,))
            fat_tmp = np.pad(np.squeeze(fat[:, slajs, :]),(patch+16,))
            gt_tmp = np.pad(np.squeeze(gt[:,:, slajs, :]),((0,0), (patch+16,patch+16), (patch+16,patch+16)))
            x_tmp = np.pad(np.squeeze(x[:, slajs, :]),(patch+16,))
            y_tmp = np.pad(np.squeeze(y[:, slajs, :]),(patch+16,))
            maska_tmp = np.squeeze(maska[:, slajs, :])

            for counter in range(to_cut):
                index = random.choice(np.argwhere(maska_tmp==1))
                startx = index[0]+16
                endx = index[0]+16+2*patch+1
                starty = index[1]+16
                endy = index[1]+16+2*patch+1

                allin = [wat_tmp[startx:endx, starty:endy], 
                        fat_tmp[startx:endx, starty:endy]]
        
                if add_dts:
                    allin.append(x_tmp[startx:endx, starty:endy])
                    allin.append(y_tmp[startx:endx, starty:endy])

                allin = np.stack(allin, axis=0)
                gt_part = gt_tmp[:, startx:endx, starty:endy]
        
                np.save(f"{outpath}in1/subj{PID}_{slajs}_{counter}", allin)
                np.save(f"{outpath}gt/subj{PID}_{slajs}_{counter}", gt_part)

                if make_subsampled:
                    startx = startx-16
                    endx = endx+16
                    starty = starty-16
                    endy = endy+16

                    allin = [wat_tmp[startx:endx:3, starty:endy:3], 
                            fat_tmp[startx:endx:3, starty:endy:3]]
                    if add_dts:
                        allin.append(x_tmp[startx:endx:3, starty:endy:3])
                        allin.append(y_tmp[startx:endx:3, starty:endy:3])
                    allin = np.stack(allin, axis=0)

                    np.save(f"{outpath}in2/subj{PID}_{slajs}_{counter}", allin) 


#cut()
# %%
