#%%
import numpy as np
import nibabel as nib 
from glob import glob
import pathlib
import random
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

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

def check2Dcuts(datafolder, pid, inp2=False):
    findit = glob(f"./{datafolder}/*/*/subj{pid}.npy")
    findit.sort()
    in1 = np.load(findit[1])
    if inp2:
        in2 = np.load(findit[2])
    gt = flatten_one_hot(np.load(findit[0]))

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
#check2Dcuts("POEM110", "500077_30_0", True)
#check2Dcuts("POEM110", "500077_30_1", True)
#check2Dcuts("POEM110", "500297_30_0", True)
#check2Dcuts("POEM110", "500297_30_1", True)
####################################


#%%
def cutPOEM(patch_size, make_subsampled, add_dts, outpath, sampling=None):
    #prepare folders for saving:
    outpath = f"{outpath}/TRAIN"
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    for i in ['gt','in1','in2']:
        pathlib.Path(f"{outpath}/{i}").mkdir(parents=True, exist_ok=True)

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

    nb_class = 7
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

        inx = wat.shape[0]//patch_size
        iny = wat.shape[2]//patch_size
        to_cut = 2 #max(4, inx*iny)

        if sampling==None:
            #if no sampling given, we cut randomly a few (to_cut=2?) patches from EACH slice.
            dict_tmp = { s: [random.choice(np.argwhere(maska[:,s,:]==1)) for i in range(to_cut)] for s in range(wat.shape[1]) }

        else: 
            assert len(sampling)==nb_class, f"Sampling variable should be an array of length 7!"
            #let's make a dict of all the slices(keys) and indeces (value lists)
            dict_tmp = {}
            for organ, nr_samples in enumerate(sampling):
                possible = np.argwhere( (gt[organ, ...]*maska) == 1)
                Ll = len(possible)
                nr_sample = min(nr_samples, Ll)
                samp = random.sample(range(Ll), nr_sample)
                samples = possible[samp, ... ]

                for onesample in samples:
                    if onesample[1] not in dict_tmp:
                        dict_tmp[onesample[1]] = []
                    dict_tmp[onesample[1]].append([onesample[0], onesample[2]])
            
        for slajs, indexes in tqdm( dict_tmp.items() ):

            wat_tmp = np.pad(np.squeeze(wat[:, slajs, :]),(patch+16,))
            fat_tmp = np.pad(np.squeeze(fat[:, slajs, :]),(patch+16,))
            gt_tmp = np.pad(np.squeeze(gt[:,:, slajs, :]),((0,0), (patch+16,patch+16), (patch+16,patch+16)))
            x_tmp = np.pad(np.squeeze(x[:, slajs, :]),(patch+16,))
            y_tmp = np.pad(np.squeeze(y[:, slajs, :]),(patch+16,))

            for counter,index in enumerate(indexes):
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
        
                np.save(f"{outpath}/in1/subj{PID}_{slajs}_{counter}", allin)
                np.save(f"{outpath}/gt/subj{PID}_{slajs}_{counter}", gt_part)

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

                    np.save(f"{outpath}/in2/subj{PID}_{slajs}_{counter}", allin) 
    if sampling!=None:
        #save sampling strategy info 
        with open(f"{outpath}/sampling.txt", 'w') as out_file:
            out_file.write(str(sampling))




def train_val_splitPOEM(datafolder, val_subjects = 15):
    #randomly chooses val_subject pids, and moves corresponding patches to VAL datafolder. 

    #prepare folders for saving:
    outpath = f"{datafolder}/VAL"
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    for i in ['gt','in1','in2']:
        pathlib.Path(f"{outpath}/{i}").mkdir(parents=True, exist_ok=True)

    #find all pids that should be in TRAIN atm
    all_paths = [p for p in pathlib.Path(datafolder).glob("**/*.npy")]
    pids = np.random.choice(np.unique([re.findall(r"500[0-9]+", fil.name)[0] for fil in all_paths]), size=val_subjects, replace=False)
    to_move = [fil for fil in all_paths if re.findall(r"500[0-9]+", fil.name)[0] in pids]
    for fil in to_move:
        fil.replace(pathlib.Path(re.sub("TRAIN", "VAL", str(fil))))


# %%
outpath = "POEM110"
add_dts = False
make_subsampled = True
patch_size = 110

#cutPOEM(patch_size, make_subsampled, add_dts, outpath)
#now all cut imges are saved in training. If you want also val data, use POEM_train_val_split:
#train_val_splitPOEM('POEM110', 15)
# %%
outpath = "POEM_sampled"
add_dts = False
make_subsampled = True
patch_size = 110
sampling = [5, 8, 10, 6, 15, 15, 10]

#cutPOEM(patch_size, make_subsampled, add_dts, outpath, sampling)
# %%
#train_val_splitPOEM('POEM_sampled', 15)

# %%
