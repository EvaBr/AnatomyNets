#%%
import numpy as np
import nibabel as nib 
from glob import glob
import pathlib
import random
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt as dt_edt
from helpers import get_one_hot, flatten_one_hot


#%%
############################  CHECKING CUTS #########################

def check2Dcuts(datafolder, pid, chans=0, inp2=False):
    findit = glob(f"./{datafolder}/*/*/*{pid}.npy")
    findit.sort()
    in1 = np.load(findit[1])
    if inp2:
        in2 = np.load(findit[2])
    gt = flatten_one_hot(np.load(findit[0]))

    vseh = 2 + 1*inp2
    plt.figure(figsize=(20,7))
    plt.subplot(1,vseh,1)
    plt.imshow(gt, vmin=0, vmax=7)
    plt.axis('off')
    plt.subplot(1,vseh,2)
    plt.imshow(np.squeeze(in1[chans,...]))
    plt.axis('off')
    if inp2:
        plt.subplot(1,vseh,3)
        plt.imshow(np.squeeze(in2[chans,...]))
        plt.axis('off')
    plt.show()


def check3Dcuts(datafolder, pid, chans=0, inp2=False):
    if isinstance(pid, str) and pid[-4:]!='.npy':
        pid = list(set(p.name for p in pathlib.Path(datafolder).glob(f"**/*{pid}*")))
    if isinstance(pid, list):
        for p in pid:
            check3Dcuts(datafolder, p, chans=chans, inp2=inp2)
        return None
    findit = glob(f"./{datafolder}/*/*/*{pid}*")
    findit.sort()
    in1 = np.load(findit[1])
    if inp2:
        in2 = np.load(findit[2])
    gt = flatten_one_hot(np.load(findit[0]))

    midx, midy, midz = gt.shape
    vsehrows = 2 + inp2*1
    plt.figure(figsize=(20,7*vsehrows))
    #plot GT slices
    plt.subplot(vsehrows,3,1)
    plt.title('gt')
    plt.imshow(gt[midx//2,...], vmin=0, vmax=7)
    plt.axis('off')
    plt.subplot(vsehrows,3,2)
    plt.imshow(gt[:,midy//2,...], vmin=0, vmax=7)
    plt.axis('off')
    plt.subplot(vsehrows,3,3)
    plt.imshow(gt[..., midz//2], vmin=0, vmax=7)
    plt.axis('off')
    #plt input
    plt.subplot(vsehrows,3,4)
    plt.title('in1')
    plt.imshow(np.squeeze(in1[chans,midx//2,...]))
    plt.axis('off')
    plt.subplot(vsehrows,3,5)
    plt.imshow(np.squeeze(in1[chans,:,midy//2,:]))
    plt.axis('off')
    plt.subplot(vsehrows,3,6)
    plt.imshow(np.squeeze(in1[chans,...,midz//2]))
    plt.axis('off')

    if inp2:
        plt.subplot(3,3,7)
        plt.title('in2')
        plt.imshow(np.squeeze(in2[chans,midx//2,...]))
        plt.axis('off')
        plt.subplot(3,3,8)
        plt.imshow(np.squeeze(in2[chans,:,midy//2,...]))
        plt.axis('off')
        plt.subplot(3,3,9)
        plt.imshow(np.squeeze(in2[chans,..., midz//2]))
        plt.axis('off')
    plt.show()
   


#%%
############################  CUTTING #########################

def cutPOEM2D(patch_size, outpath, make_subsampled=True, add_dts=True, sliced=1, sampling=None):
    #sliced je lahko 0,1 ali 2. pove po katerem indexu naredimo slice. 
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
    assert len(gt_paths)==len(wat_paths)==len(fat_paths)==len(dtx_paths)==len(dty_paths)==len(mask_paths)

    nb_class = 7
    patch = patch_size//2

    slicing = ":,"*sliced + "slajs" + ",:"*(2-sliced) + "]"  
    print(f"\nSLICING: [{slicing}\n")
    for w,f,g,dx,dy,m in zip(wat_paths, fat_paths, gt_paths, dtx_paths, dty_paths, mask_paths):
        PIDs = [re.findall(r"500[0-9]+", ppp)[0] for ppp in [w,f,g,dx,dy,m]]
        assert len(np.unique(PIDs)) == 1
        PID = PIDs[0]
        print(f"Slicing nr {PID}...")
        wat = nib.load(w).get_fdata()
        fat = nib.load(f).get_fdata()
        gt = nib.load(g).get_fdata()
        x = nib.load(dx).get_fdata()
        y = nib.load(dy).get_fdata()
        maska = nib.load(m).get_fdata()

        tmp_z = np.ones(maska.shape)
        startz, endz = np.nonzero(maska.sum(axis=(0,1)))[0][0], np.nonzero(maska.sum(axis=(0,1)))[0][-1]
        tmp_z[:,:,startz] = 0
        tmp_z = 2.*dt_edt(tmp_z)/(endz-startz) - 1.

        z = maska*tmp_z#create artificially, simply DT from left to right
        bd = dt_edt(maska) #create artificially, simply DT from border
        bd = bd/np.max(bd)

        gt = get_one_hot(gt, nb_class) #new size C x H x W x D

        inx = wat.shape[1-(sliced>0)]//patch_size
        iny = wat.shape[2-(sliced==2)]//patch_size
        to_cut = 2 #max(4, inx*iny)

        dict_tmp = {}

        if sampling==None:
            #if no sampling given, we cut randomly a few (to_cut=2?) patches from EACH slice.
           # print((maska.shape, wat.shape))
            for slajs in range(maska.shape[sliced]):
                kjeso = eval(f"np.argwhere(maska[{slicing}==1)")
                if len(kjeso)>to_cut:
                    dict_tmp[slajs] = [random.choice(kjeso) for i in range(to_cut)]

        else: 
            assert len(sampling)==nb_class, f"Sampling variable should be an array of length 7!"
            #let's make a dict of all the slices(keys) and indeces (value lists)
            for organ, nr_samples in enumerate(sampling):
                possible = np.argwhere( (gt[organ, ...]*maska) == 1)
                Ll = len(possible)
                nr_sample = min(nr_samples, Ll)
                samp = random.sample(range(Ll), nr_sample)
                samples = possible[samp, ... ]

                for onesample in samples:
                    if onesample[1] not in dict_tmp:
                        dict_tmp[onesample[sliced]] = []
                    dict_tmp[onesample[sliced]].append([onesample[left] for left in range(3) if left!=sliced])
            
        for slajs, indexes in tqdm( dict_tmp.items() ):

            wat_tmp = np.pad(np.squeeze(eval(f"wat[{slicing}")),(patch+16,))
            fat_tmp = np.pad(np.squeeze(eval(f"fat[{slicing}")),(patch+16,))
            gt_tmp = np.pad(np.squeeze(eval(f"gt[:,{slicing}")),((0,0), (patch+16,patch+16), (patch+16,patch+16)))
            x_tmp = np.pad(np.squeeze(eval(f"x[{slicing}")),(patch+16,))
            y_tmp = np.pad(np.squeeze(eval(f"y[{slicing}")),(patch+16,))
            z_tmp = np.pad(np.squeeze(eval(f"z[{slicing}")),(patch+16,))
            bd_tmp = np.pad(np.squeeze(eval(f"bd[{slicing}")),(patch+16,))

            for counter,index in enumerate(indexes):
                startx = index[0]+16
                endx = index[0]+16+2*patch
                starty = index[1]+16
                endy = index[1]+16+2*patch

                allin = [wat_tmp[startx:endx, starty:endy], 
                        fat_tmp[startx:endx, starty:endy]]
        
                if add_dts:
                    allin.append(x_tmp[startx:endx, starty:endy])
                    allin.append(y_tmp[startx:endx, starty:endy])
                    allin.append(z_tmp[startx:endx, starty:endy])
                    allin.append(bd_tmp[startx:endx, starty:endy])

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
                        allin.append(z_tmp[startx:endx:3, starty:endy:3])
                        allin.append(bd_tmp[startx:endx:3, starty:endy:3])
                    allin = np.stack(allin, axis=0)

                    np.save(f"{outpath}/in2/subj{PID}_{slajs}_{counter}", allin) 
    if sampling!=None:
        #save sampling strategy info 
        with open(f"{outpath}/sampling.txt", 'w') as out_file:
            out_file.write(str(sampling))
    with open(f"{outpath}/datainfo.txt", "w") as info_file:
        info_file.write(f"Sliced by dim {sliced}. \nPatch size: {patch_size}\nDTs: {add_dts}\nsubsmpl: {make_subsampled}")

#%%
def cutPOEM3D(patch_size, outpath, make_subsampled=True, add_dts=True, sampling=None):
    #sampling pove koliko patchov per class samplamo iz vsakega subjekta. 
    # If not given, sampling is random. (ie may contain lots of bckg!) 

    #prepare folders for saving:
    outpath = f"{outpath}/TRAIN"
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    for i in ['gt','in1','in2']:
        pathlib.Path(outpath,i).mkdir(parents=True, exist_ok=True)

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
    assert len(gt_paths)==len(wat_paths)==len(fat_paths)==len(dtx_paths)==len(dty_paths)==len(mask_paths)

    nb_class = 7
    patch = patch_size//2
 
    for w,f,g,dx,dy,m in zip(wat_paths, fat_paths, gt_paths, dtx_paths, dty_paths, mask_paths):
        PIDs = [re.findall(r"500[0-9]+", ppp)[0] for ppp in [w,f,g,dx,dy,m]]
        assert len(np.unique(PIDs)) == 1
        PID = PIDs[0]
        print(f"Slicing nr {PID}...")
        wat = nib.load(w).get_fdata()
        fat = nib.load(f).get_fdata()
        gt = nib.load(g).get_fdata()
        x = nib.load(dx).get_fdata()
        y = nib.load(dy).get_fdata()
        maska = nib.load(m).get_fdata()

        tmp_z = np.ones(maska.shape)
        startz, endz = np.nonzero(maska.sum(axis=(0,1)))[0][0], np.nonzero(maska.sum(axis=(0,1)))[0][-1]
        tmp_z[:,:,startz] = 0
        tmp_z = 2.*dt_edt(tmp_z)/(endz-startz) - 1.

        z = maska*tmp_z#create artificially, simply DT from left to right
        bd = dt_edt(maska) #create artificially, simply DT from border
        bd = bd/np.max(bd)

        gt = get_one_hot(gt, nb_class) #new size C x H x W x D

        inx = wat.shape[0]//patch_size
        iny = wat.shape[1]//patch_size
        inz = wat.shape[2]//patch_size
        to_cut = 5 


        if sampling==None:
            #if no sampling given, we cut randomly a few (to_cut=2?) patches from EACH slice.
           # print((maska.shape, wat.shape))
            kjeso = np.argwhere(maska==1)
            if len(kjeso)>to_cut:
                kjeso = kjeso[np.random.choice(kjeso.shape[0], to_cut, replace=False), ...]

        else: 
            assert len(sampling)==nb_class, f"Sampling variable should be an array of length 7!"
            #let's make a dict of all the slices(keys) and indeces (value lists)
            kjeso = []
            for organ, nr_samples in enumerate(sampling):
                possible = np.argwhere( (gt[organ, ...]*maska) == 1)
                Ll = len(possible)
                nr_sample = min(nr_samples, Ll)
                kjeso.append(possible[random.sample(range(Ll), nr_sample),...])
            kjeso = np.vstack(kjeso)
                
                
               
        wat_tmp = np.pad(wat,(patch+16,))
        fat_tmp = np.pad(fat,(patch+16,))
        gt_tmp = np.pad(gt,((0,0), (patch+16,patch+16), (patch+16,patch+16),(patch+16,patch+16)))
        x_tmp = np.pad(x,(patch+16,))
        y_tmp = np.pad(y,(patch+16,))
        z_tmp = np.pad(z,(patch+16,))
        bd_tmp = np.pad(bd,(patch+16,))

        for idx, center in tqdm( enumerate(kjeso) ):
            startx = center[0]+16
            endx = center[0]+16+2*patch
            starty = center[1]+16
            endy = center[1]+16+2*patch
            startz = center[2]+16
            endz = center[2]+16+2*patch

            allin = [wat_tmp[startx:endx, starty:endy, startz:endz], 
                    fat_tmp[startx:endx, starty:endy, startz:endz]]
        
            if add_dts:
                allin.append(x_tmp[startx:endx, starty:endy, startz:endz])
                allin.append(y_tmp[startx:endx, starty:endy, startz:endz])
                allin.append(z_tmp[startx:endx, starty:endy, startz:endz])
                allin.append(bd_tmp[startx:endx, starty:endy, startz:endz])

            allin = np.stack(allin, axis=0)
            gt_part = gt_tmp[:, startx:endx, starty:endy, startz:endz]
        
            np.save(f"{outpath}/in1/subj{PID}_{idx}_0", allin)
            np.save(f"{outpath}/gt/subj{PID}_{idx}_0", gt_part)

            if make_subsampled:
                startx = startx-16
                endx = endx+16
                starty = starty-16
                endy = endy+16
                startz = startz-16
                endz = endz+16

                allin = [wat_tmp[startx:endx:3, starty:endy:3, startz:endz:3], 
                        fat_tmp[startx:endx:3, starty:endy:3, startz:endz:3]]
                if add_dts:
                    allin.append(x_tmp[startx:endx:3, starty:endy:3, startz:endz:3])
                    allin.append(y_tmp[startx:endx:3, starty:endy:3, startz:endz:3])
                    allin.append(z_tmp[startx:endx:3, starty:endy:3, startz:endz:3])
                    allin.append(bd_tmp[startx:endx:3, starty:endy:3, startz:endz:3])
                allin = np.stack(allin, axis=0)

                np.save(f"{outpath}/in2/subj{PID}_{idx}_0", allin) 

    if sampling!=None:
        #save sampling strategy info 
        with open(f"{outpath}/sampling.txt", 'w') as out_file:
            out_file.write(str(sampling))
    with open(f"{outpath}/datainfo.txt", "w") as info_file:
        info_file.write(f"Sliced 3D patches. \nPatch size: {patch_size}\nDTs: {add_dts}\nsubsmpl: {make_subsampled}")


#cutting 2d whole slices, as in BL project
def cutPOEMslices():
    #by default cuts only in axial direction. This was just for tryouts; same date as in BL project.
    outpath = f"POEM_slices/TRAIN"
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    for i in ['gt','in1','in2']:
        pathlib.Path(f"POEM_slices/{i}").mkdir(parents=True, exist_ok=True)

    #POEM SLICING
    gt_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segment_all/converted/CroppedSegmNew*")
    wat_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/cropped*_wat*")
    fat_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/cropped*_fat*")
    dtx_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/distmaps/*x.nii")
    dty_paths = glob("/home/eva/Desktop/research/PROJEKT2-DeepLearning/distmaps/*y.nii")
   
    gt_paths.sort()
    wat_paths.sort()
    fat_paths.sort()
    dtx_paths.sort()
    dty_paths.sort()
   
    for w,f,g,dx,dy in zip(wat_paths, fat_paths, gt_paths, dtx_paths, dty_paths):
        PID = re.findall(r"500[0-9]+", w)[0]
        print(f"Slicing nr {PID}...")
        wat = nib.load(w).get_fdata()
        fat = nib.load(f).get_fdata()
        gt = nib.load(g).get_fdata()
        x = nib.load(dx).get_fdata()
        y = nib.load(dy).get_fdata()
        
        gt = get_one_hot(gt, 7) #new size C x H x W x D

        slajsi_where = gt[1:, ...].sum(axis=(0,1,3))
        slajsi = np.arange(wat.shape[1])
        slajsi = slajsi[slajsi_where>0]

        for slajs in tqdm( slajsi ):
            allin = [wat[:, slajs, ...], fat[:, slajs, ...], x[:,slajs,...], y[:,slajs,...]]
            allin = np.stack(allin, axis=0)
            quasidownsmp = allin[:, 0::3, 0::3]
            gt_part = gt[:, :,  slajs, :]
        
            np.save(f"POEM_slices/in1/subj{PID}_{slajs}_0", allin)
            np.save(f"POEM_slices/in2/subj{PID}_{slajs}_0", quasidownsmp)
            np.save(f"POEM_slices/gt/subj{PID}_{slajs}_0", gt_part)


    

#%% 

############################  POSTPROCESS CUTS  #########################
#(works independent of dimensions!!)
def train_val_splitPOEM(datafolder, val_subjects = 15):
    #randomly chooses val_subject pids, and moves corresponding patches to VAL datafolder. 
    #check first if TRAIN exists:
    if not pathlib.Path(datafolder, "TRAIN").exists():
        pathlib.Path(datafolder, "TRAIN").mkdir(parents=True)
        for pp in ['gt', 'in1', 'in2']:
            pathlib.Path(datafolder, pp).rename(pathlib.Path(datafolder, "TRAIN", pp))

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



#perhaps training using only slices with at least one other class than background would be better?
#let's remove all 0 slices for now
def remove_bckg_slices(datafolder):
    all_paths = [p for p in pathlib.Path(datafolder).glob("*gt/*.npy")] #we need ground truth to know
    for p in tqdm(all_paths):
        pid = re.findall(r"500[0-9]+_[0-9]+_[0-9]+", p.name)[0]
        fil = np.load(p)
        classes = fil.sum(axis=tuple([i for i in range(1, fil.ndim)])) #sum by all axes but 0th(=classes)
        if classes[1:].sum() == 0: #no other class but 0 present, delete gt and in1 and in2. 
            p.unlink()
            for f in pathlib.Path(datafolder).glob(f"in*/*{pid}.npy"):
                f.unlink()


# %%
#example runs:

#outpath = "POEM80"
#add_dts = False
#make_subsampled = False
#patch_size = 80

#cutPOEM2D(patch_size, outpath, make_subsampled=make_subsampled, add_dts=add_dts)
#train_val_splitPOEM('POEM80', 15)

#remove_bckg_slices('POEM80/TRAIN')
#remove_bckg_slices('POEM80/VAL')


#check2Dcuts("POEM80", "500077_30_0", True)

# %%
