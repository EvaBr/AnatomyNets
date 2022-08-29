#%%
from pathlib import Path
from tqdm import tqdm
import numpy as np 
import Networks
import torch
from glob import glob
from helpers import flatten_one_hot, get_one_hot, CenterCropTensor, CenterCropTensor3d
from Losses import subjectDices
import nibabel as nib
from typing import List, Tuple, Dict, Any, Union
from scipy.ndimage import distance_transform_edt as dt_edt
from slicer import Slicer
#%%


def getchn(args: List[str], string:str) -> Tuple[int, List[int]]:
    cnt = 0
    whichchans = []
    start = [i for i in range(len(args)) if string in args[i]]
    if len(start)==0: #not given, using default channels 0,1
        return 2, [0,1]

    for i in range(start[0]+1, len(args)):
        if '--' in args[i]:
            break 
        cnt+=1
        whichchans.append(int(args[i]))
    return cnt, whichchans


def getNetwork(params:str, dev:str = 'cpu', best:bool=True) -> Tuple[Any, List, List, bool]:
    #default settings:
    Arg = {'network': None, 'n_class':7, 'in_channels':2, 'lower_in_channels':2, 'extractor_net':'resnet34'}
    
    with open(f"RESULTS/{params}_args.txt", "r") as ft: #TODO: fix to pathlib for easier use
        args = ft.read().splitlines()
    tmpargs = [i.strip('--').split("=") for i in args if ('--' in i and '=' in i)]
    chan1, whichin1 = getchn(args, 'in_chan')
    chan2, whichin2 = getchn(args, 'lower_in_chan')
    tmpargs += [['in_channels', chan1], ['lower_in_channels', chan2]]
    in3D='--in3D' in args
    args = dict(tmpargs)

    #overwrite if given in file:
    Arg.update(args)
    use_in2 = Arg['network']=='DeepMedic'

    net = getattr(Networks, Arg['network'])(Arg['in_channels'], Arg['n_class'], Arg['lower_in_channels'], Arg['extractor_net'], in3D)
    net = net.float()
    #now we can load learned params:
    if best:
        params = params+'_bestepoch'
    loaded = torch.load(f"RESULTS/{params}", map_location=lambda storage, loc: storage)
    net.load_state_dict(loaded['state_dict'])
    
    device = torch.device(dev)
    net = net.to(device)

    return net, whichin1, whichin2 if use_in2 else [], in3D


def loadSubject(pid:int, leavebckg:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #leavebckg==what border of bck to leave around subj (otherwise we cut the imgs so that 
    # they are as tight around subj (mask==1) as possible).
    dx, dy, m, g, f, w = sorted(Path('POEM').rglob(f"*{pid}*.nii"))
    # order: dtx, dty, mask, gt, fat, wat
    wat = nib.load(str(w)).get_fdata()
    fat = nib.load(str(f)).get_fdata()
    gt = nib.load(str(g)).get_fdata()
    maska = nib.load(str(m)).get_fdata()
    x = nib.load(str(dx)).get_fdata()
    y = nib.load(str(dy)).get_fdata()       
    gt = get_one_hot(gt*maska, 7) #to make sure segms will only be done inside subj, lets multiply by mask:
   
    tmp_z = np.ones(maska.shape)
    tmp = maska.sum(axis=(0,1))
    startz, endz = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
    tmp_z[:,:,startz] = 0
    tmp_z = 2.*dt_edt(tmp_z)/(endz-startz) - 1.

    z = maska*tmp_z#create artificially, simply DT from left to right
    bd = dt_edt(maska) #create artificially, simply DT from border
    bd = bd/np.max(bd)

    allin = np.stack([wat, fat, x, y, z, bd], axis=0)
    
    tmp = maska.sum(axis=(1,2))
    startx, endx = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
    tmp = maska.sum(axis=(0,2))
    starty, endy = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
    
    #new starts/ends based on the required border width:
    x, y, z = maska.shape
    startx = max(0, startx-leavebckg) 
    starty = max(0, starty-leavebckg) 
    startz = max(0, startz-leavebckg) 
    endx = min(x, endx+leavebckg+1)
    endy = min(y, endy+leavebckg+1)
    endz = min(z, endz+leavebckg+1)

   # print(("orig.sizes:", maska.shape))
   # print(("new slice:", (startx,endx,starty,endy,startz,endz)))
    maska = maska[startx:endx, starty:endy, startz:endz]
    allin = allin[:, startx:endx, starty:endy, startz:endz]
   # print(("new sizes:", maska.shape, allin.shape))
    #to make sure segms will only be done inside subj, lets multiply by mask:
    return allin*maska, gt[:, startx:endx, starty:endy, startz:endz], maska
    

#%%
def Compute3DDice(PID:Union[int, List[int]], netparams:str, patchsize:int, 
            batch:int = 10, bydim:int = 1, doeval:bool = True, 
            dev:str = 'cpu', best:bool=True, step:int=0, saveout:bool=False, savename:str='x') -> List[float]:
    #OBS: in case of deepmed, patchsize means the size of output patch!
    #(i.e. if patchsize=9, the input to network will be 25x25) <-but this done in the code
    #step = in what steps you take patches. if ==0, you take nonoverlapping ones. if K, patch starts 
    # at prev_patch_start+K.
    #saveout = whether we save the ful subject output. (for viewing and debugging)

    # GET NET:
    net, in1, in2, in3D = getNetwork(netparams, dev, best)
    if doeval:
        net.eval()
    else:
        net.train()
    device = torch.device(dev)
    print(f'Net loaded. [in1: {in1}, in2: {in2}, 3D: {in3D}]')
    # CUT AND EVAL: loop through cutting smaller pieces, moving to torch and eval
    if isinstance(PID, int): 
        PID=[PID]
    segmented = torch.zeros((1,7), device=dev)
    existing = torch.zeros((1,7), device=dev)
    intersec = torch.zeros((1,7), device=dev) #these three needed to gather results, for post dice compute
    Dices = torch.zeros((len(PID), 7), device=dev)
    axes = [0,2,3] + ([4] if in3D else [])

    #set the right function to use
    TensorCropping = CenterCropTensor3d
    padding = [(0,0),(0,patchsize),(0,patchsize),(0,patchsize)] #(16,patchsize+16)
    paddingall = [(0,0),(0,patchsize),(0,patchsize),(0,patchsize)] #(16,patchsize+16)
    
    if in2: #deep med, we need to pad the input on all sides to be able to cut pieces as wanted
        paddingall[1:] = [(16+8,patchsize+16+8)]*3
        patchsize = patchsize-16 
        #since patchsize, as it goes into slicer, means the size of network output
        
    if not in3D: 
        padding[bydim+1] = (0,0)
        paddingall[bydim+1] = (0,0)
        TensorCropping = CenterCropTensor

    # LOAD DATA:
    for idx, pid in enumerate(PID):
        #set accumulators to 0:
        segmented.zero_()
        existing.zero_()
        intersec.zero_()

        allin, gt, mask = loadSubject(pid, patchsize//2)
    
        size_full = allin[0].shape #shape of 3d img, one channel
    
        mask = np.pad(mask, padding[1:], mode='constant')
        gt = np.pad(gt, padding, mode='constant')
        allin = np.pad(allin, paddingall, mode='constant')
      #  print((size_full, gt.shape))
        empty_subj = torch.zeros(gt.shape[1:]) #allin.shape[1:]) #cause we dont need channels
        
        slicer = Slicer(size_full, patchsize, in1, in2, in3D, bydim, step) #return string slice, include all channels
        # for cutting out the middle part based on step:
        #slice((sf-step)//2, sf-np.ceil((sf-step)/2))
        slicing = "".join([f'.narrow({idx}, {(patchsize-step)//2}, {step})' for idx in range(2,(4+in3D))]) if step>0 else ""
        paddingup = [0,patchsize-step]*3
        if not in3D:
            paddingup[-1-bydim*2] = 0

        print(f'Eval on subj{pid}...')
        with torch.no_grad():
            while slicer.todo>0:
                gtslices, in1slices, in2slices =  slicer.get_batch(batch) #multiple slices
        
                gts = np.stack(list(map(eval, [f'gt[{slajs}]' for slajs in gtslices])), axis=0)
                in1s = np.stack(list(map(eval, [f'allin[{slajs}]' for slajs in in1slices])), axis=0)
                #maske = np.stack([eval(f'mask[{slajs[2:]}]') for slajs in gtslices], axis=0)
                maske = np.stack(list(map(eval, [f'mask[{slajs[2:]}]' for slajs in gtslices])), axis=0)
            
                # move to torch:
                target_oh = torch.from_numpy(gts).squeeze().to(device)
                data = [torch.from_numpy(in1s).squeeze().float().to(device)] #input 1
                if in2:
                    #in2s = np.stack([eval(f'allin[{slajs}]') for slajs in in2slices], axis=0)
                    in2s = np.stack(list(map(eval, [f'allin[{slajs}]' for slajs in in2slices])), axis=0)
                    data.append(torch.from_numpy(in2s).squeeze().float().to(device)) #input 2
    
                #run net on data. get output, save sums in dice gather lists
                out = net(*data).exp()
                target_oh, out = TensorCropping(target_oh, out) #in case of PSP net, might be that output is bigger than input/GT
                #dices = AllDices(out, target_oh)
                maske = torch.from_numpy(maske).squeeze().unsqueeze(1).float().to(device)
                
                
                #cut only the middle part of OUT, MASKE and TARGET_OH for eval (depending on the step size)
                maske = eval('maske'+slicing)
                target_oh = eval('target_oh'+slicing)
                out = eval('out'+slicing)

                #when summing up, use only the middle of the patches. Depending on how big 'step' was. 
                segmented += torch.sum(out*maske, axis=axes)
                existing += torch.sum(target_oh*maske, axis=axes)
                intersec += torch.sum(target_oh*maske*out, axis=axes)

                #save output if required
                if saveout: #whats faster, simply saving to an existing tensor, or iffing every loop??
                    for idd, slajs in enumerate(gtslices):
                        tmp = torch.argmax(out[idd,...], dim=0)
                        if not in3D:
                            tmp = tmp.unsqueeze(bydim)
                       # print(tmp.shape)
                        tmp = torch.nn.functional.pad(tmp, paddingup)
                        eval(f'empty_subj[{slajs[2:]}].copy_(tmp)')

                
        #all saved, now calc actual dices:
        Dices[idx, :] = 2*intersec/(existing+segmented) #calc dice from the gathering lists
        if saveout:
            #save img as npy.
            np.save(f'out{pid}_{savename}.npy', empty_subj.cpu().numpy())

    print('Done.')
    #pidis = [int(p) for p in PIDS]
    dices = np.concatenate((np.array(PID)[:,None], Dices.cpu().numpy()), axis=1)
    np.save(f'dices_{savename}.npy', dices)
    return dices



# %%
