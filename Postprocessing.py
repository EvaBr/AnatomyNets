#%%
from pathlib import Path
from scipy import empty
from sqlalchemy import false, true
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
import pandas as pd
import matplotlib.pyplot as plt
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


def loadSubject(pid:int, leavebckg:int=25, cutit:bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #leavebckg==what border of bck to leave around subj (otherwise we cut the imgs so that 
    # they are as tight around subj (mask==1) as possible).
    #but if cutit=false, instead of cutting tightly around mask, we load the whole image and 
    
    dx, dy, m, g, f, w = sorted(Path('POEM').rglob(f"*{pid}*.nii"))
    # order: dtx, dty, mask, gt, fat, wat
    wat = nib.load(str(w)).get_fdata()
    fat = nib.load(str(f)).get_fdata()
    gt = nib.load(str(g)).get_fdata()
    maska = nib.load(str(m)).get_fdata()
    x = nib.load(str(dx)).get_fdata()
    y = nib.load(str(dy)).get_fdata()       
    gt = get_one_hot(gt*maska, 7) #to make sure segms will only be done inside subj, lets multiply by mask:
   
    #if not cutit:
        # print(("orig.sizes:", maska.shape))
    #    maska  = np.pad(maska, leavebckg, 'edge')
    #    gt  = np.pad(gt, ((0,0),(leavebckg, leavebckg),(leavebckg, leavebckg),(leavebckg,leavebckg )), 'edge')
    #    x  = np.pad(x, leavebckg, 'edge')
    #    y  = np.pad(y, leavebckg, 'edge')
    #    wat  = np.pad(wat, leavebckg, 'edge')
    #    fat  = np.pad(fat, leavebckg, 'edge')
        # print(("new.sizes:", maska.shape))

    tmp_z = np.ones(maska.shape)
    tmp = maska.sum(axis=(0,1))
    startz, endz = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
    tmp_z[:,:,startz] = 0
    tmp_z = 2.*dt_edt(tmp_z)/(endz-startz) - 1.

    z = maska*tmp_z#create artificially, simply DT from left to right
    bd = dt_edt(maska) #create artificially, simply DT from border
    bd = bd/np.max(bd)

    allin = np.stack([wat, fat, x, y, z, bd], axis=0)
    
    
    if cutit:   
        #new starts/ends based on the required border width:
        x, y, z = maska.shape
        tmp = maska.sum(axis=(1,2))
        startx, endx = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
        tmp = maska.sum(axis=(0,2))
        starty, endy = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
    
        startx = max(0, startx-leavebckg) 
        starty = max(0, starty-leavebckg) 
        startz = max(0, startz-leavebckg) 
        endx = min(x, endx+leavebckg+1)
        endy = min(y, endy+leavebckg+1)
        endz = min(z, endz+leavebckg+1)
        # print(("orig.sizes:", maska.shape))
        #print(f"new slice: ({startx}:{endx},{starty}:{endy},{startz}:{endz})")
        maska = maska[startx:endx, starty:endy, startz:endz]
        allin = allin[:, startx:endx, starty:endy, startz:endz]
        gt = gt[:, startx:endx, starty:endy, startz:endz]
        # print(("new sizes:", maska.shape, allin.shape))

   
   
    #to make sure segms will only be done inside subj, lets multiply by mask:
    return allin*maska, gt, maska
    

#%%
def Compute3DDice(PID:Union[int, List[int]], netparams:str, patchsize:int, 
            batch:int = 32, bydim:int = 1, doeval:bool = True, 
            dev:str = 'cuda', best:bool=True, step:int=0, saveout:bool=False, savename:str='x') -> List[float]:
    #OBS: in case of deepmed, patchsize means the size of output patch!
    #(i.e. if patchsize=9, the input to network will be 25x25) <-but this done in the code
    #step = in what steps you take patches. if ==0, you take nonoverlapping ones. if K, patch starts 
    # at prev_patch_start+K.
    #saveout = whether we save the ful subject output. (for viewing and debugging)

    bestorlast='LAST'
    if best:
        bestorlast='BEST'
    # GET NET:
    net, in1, in2, in3D = getNetwork(netparams, dev, best)
    if doeval:
        net.eval()
    else:
        net.train()
    device = torch.device(dev)
    print(f'Net loaded. [in1: {in1}, in2: {in2}, 3D: {in3D}]')
    # CUT AND EVAL: loop through cutting smaller pieces, moving to torch and eval
    if not isinstance(PID, list): 
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

        allin, gt, mask = loadSubject(pid) #, patchsize//2)
    
        size_full = allin[0].shape #shape of 3d img, one channel
    
        mask = np.pad(mask, padding[1:], mode='edge')
        gt = np.pad(gt, padding, mode='edge')      
        allin = np.pad(allin, paddingall, mode='edge')
      #  print((size_full, gt.shape))
        empty_subj = torch.zeros(gt.shape[1:]) #allin.shape[1:]) #cause we dont need channels
        
        slicer = Slicer(size_full, patchsize, in1, in2, in3D, bydim, step) #return string slice, include all channels
        # for cutting out the middle part based on step:
        #slice((sf-step)//2, sf-np.ceil((sf-step)/2))
        halfpad = (patchsize-step)//2
        slicing = "".join([f'.narrow({idx}, {halfpad}, {step})' for idx in range(2,(4+in3D))]) if step>0 else ""
        paddingup = [0,patchsize-step]*3 if step>0 else [0,0]*3
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
               # print(f"sizes before: gt {target_oh.shape}, out {out.shape}")
                target_oh, out = TensorCropping(target_oh, out) #in case of PSP net, might be that output is bigger than input/GT
                #dices = AllDices(out, target_oh)
               # print(f"sizes after: gt {target_oh.shape}, out {out.shape}")
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
                       # print((tmp.shape, slajs[2:]))
                       # print(paddingup)
                        
                        #tmp = torch.nn.functional.pad(tmp, paddingup)
                        nsls = re.findall(r"\d+", slajs[2:]) 
                        novslajs = f"{nsls[0]+halfpad}:{nsls[0]+halfpad+step}, {nsls[2]+halfpad}:{nsls[2]+halfpad+step}, {nsls[4]+halfpad}:{nsls[4]+halfpad+step}"

                        #eval(f'empty_subj[{slajs[2:]}].copy_(tmp)')
                        eval(f'empty_subj[{novslajs}].copy_(tmp)')

                
        #all saved, now calc actual dices:
        Dices[idx, :] = 2*intersec/(existing+segmented) #calc dice from the gathering lists
        if saveout:
            #save img as npy.
            sx, sy, sz = empty_subj.shape
            px, py, pz = [c[1] for c in padding[1:]]

            np.save(f'RESULTS/OUTS/out{pid}_{savename}_{bestorlast}.npy', empty_subj[0:sx-px, 0:sy-py, 0:sz-pz].cpu().numpy()) 
            #todo: test slicing here. 

    print('Done.')
    #pidis = [int(p) for p in PIDS]
 #   dices = np.concatenate((np.array(PID)[:,None], Dices.cpu().numpy()), axis=1)
 #   np.save(f'dices_{savename}.npy', dices)
    tabela = pd.DataFrame(Dices.cpu().numpy(), index=[PID], columns=['bckg', 'bladder', 'kidney1', 'liver', 'pancreas', 'spleen', 'kidney2'])
    tabela.to_csv(f'RESULTS/OUTS/dices_{savename}.csv', mode='a', index_label=bestorlast)
    
    return tabela

#DO :  torch.cuda.empty_cache() after each 3dDice calc!!
#%%
torch.cuda.empty_cache()

a = Compute3DDice([500056, 500304], 'un_2/unet', 25, 
            best=False, step=4, saveout=True, savename='un2')
torch.cuda.empty_cache()
print(a)


a = Compute3DDice([500056, 500304], 'un_4/unet', 25, 
            best=False, step=4, saveout=True, savename='un4')
torch.cuda.empty_cache()
print(a)


a = Compute3DDice([500056, 500304], 'un_6/unet', 25, 
            best=True, step=4, saveout=True, savename='un6')
torch.cuda.empty_cache()
print(a)


a = Compute3DDice([500056, 500304], 'un_8/unet', 25, 
            best=True, step=4, saveout=True, savename='un8')
torch.cuda.empty_cache()
print(a)

 #%%
#another way of calcng Dice per subject: (tho can only be done after CalcDice3D)
fajl = 'RESULTS/OUTS/UN-3layer/out500056_un2_LAST.npy'
out = np.load(fajl)
t=7 #set this to (patchsize-step)//2, needed for out-files made with old CalculateDice3D
s1,s2,s3=out.shape
out = out[:s1-t, :s2-t, :s3-t]
x, gt, maska = loadSubject(500056)

gt = gt[:, t:, t: , t:]
out_oh = get_one_hot(out, 7)
intersection=np.einsum("cwhd,cwhd->c", out_oh, gt)
union=(np.einsum("cwhd->c", out_oh) + np.einsum("cwhd->c", gt))

dices = (2 * intersection + 1e-10) / (union + 1e-10)



#%%
#to check best/last GDls of validation:
nr = 1
epoch = 23 #22, 24, 25, 32, 43, 36, 48
pdf = pd.read_csv(f'RESULTS/un_{nr}/unet.csv', header=0, index_col=0)
#(pdf.loc[str(epoch)]['val_GDLbin'], pdf.loc[49]['val_GDLbin'])
pdf.iloc['val_GDLbin'].idxmax()


#%%
def plotSeg(pathtofajl, slicestuple=(90,35,130), t=0):
    #fajl = 'dm1dts_v4'
    #fil = 'RESULTS/OUTS/out' + str(subjnr) + '_'+fajl+'.npy'
    fil = pathtofajl
    subjnr = re.findall("500\d+", fil)[0]

    #which slices to plot:
    s1,s2,s3 = slicestuple 
    #bydim = 0 #set to 0 when in 3d

    img = np.load(fil)
    
    gt = list(Path('POEM', 'segms').glob(f'Cropped*{subjnr}*'))[0]
    gt = nib.load(str(gt)).get_fdata()
    mask = list(Path('POEM', 'masks').glob(f'cropped*{subjnr}*'))[0]
    mask = nib.load(str(mask)).get_fdata()
    print((gt.shape, img.shape))
    #gt, img = CenterCropTensor3d(gt, img)
    #
    #gx,gy,gz = gt.shape
    #img = img[:gx,:gy,:gz]
    
    sh1,sh2,sh3=gt.shape
    img = img[:sh1-t,:sh2-t,:sh3-t]
    gt = gt[t:,t:,t:]

    plt.figure(figsize=(10,10))
    plt.subplot(3,3,1)
    plt.imshow(img[:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('axial OUT')
    plt.axis('off')
    plt.subplot(3,3,2)
    plt.imshow(img[:,s2,:].squeeze(), vmin=0, vmax=7)
    plt.title('coronal OUT')
    plt.axis('off')
    plt.subplot(3,3,3)
    plt.imshow(img[s1,:,:].squeeze(), extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('saggital OUT')
    plt.subplot(3,3,4)
    plt.imshow(gt[:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('axial GT')
    plt.axis('off')
    plt.subplot(3,3,5)
    plt.imshow(gt[:,s2,:].squeeze(), vmin=0, vmax=7)
    plt.title('coronal GT')
    plt.axis('off')
    plt.subplot(3,3,6)
    plt.imshow(gt[s1,:,:].squeeze(), extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('sagital GT')
    plt.axis('off')


    newim = np.zeros((3,sh1-t,sh2-t,sh3-t))
    newim[0,...] = gt
    newim[1,...] = img

    plt.subplot(3,3,7)
    plt.imshow(newim[:,:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('axial GT')
    plt.axis('off')
    plt.subplot(3,3,8)
    plt.imshow(np.transpose(newim[:,:,s2,:].squeeze(),(1,2,0)), vmin=0, vmax=7)
    plt.title('coronal GT')
    plt.axis('off')
    plt.subplot(3,3,9)
    plt.imshow(np.transpose(newim[:,s1,:,:].squeeze(),(1,2,0)), extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('sagital GT')
    plt.axis('off')


#%%
def compare_curves(result_folder, list_of_names, plot_names = None, individ_Dices=[0,1,2,3,4,5,6]):
#        plot_names = kako jih imenovat v plotu. If none, reuse list_of_names
#        individ_dices = which individual dices to plot the curves for (if empty, plots only GDL and Loss"""

    if plot_names==None:
        plot_names = list_of_names
    #read in metrics
    pot = Path('RESULTS', result_folder)
    #metrics = {plotname: pd.read_csv(f"RESULTS/{name}.csv") for plotname,name in zip(plot_names, list_of_names)}
    metrics = {plotname: pd.read_csv(list(Path(pot, name).rglob('*.csv'))[0]) for plotname,name in zip(plot_names, list_of_names)}
    Dice_names = ['Dice_bck','Dice_Bladder', 'Dice_KidneyL', 'Dice_Liver', 'Dice_Pancreas', 'Dice_Spleen', 'Dice_KidneyR']
    to_plot = ['Loss'] + [Dice_names[i] for i in individ_Dices]

    L = len(to_plot)
    cols, rows = min(3,L), np.ceil(L/3)
    for tip in [("", "TRAINING"), ("val_", "VALIDATION")]:
        plt.figure(figsize = (cols*7,rows*5))
        plt.suptitle(tip[1])
        for idx, what in enumerate(to_plot):
            plt.subplot(rows, cols, idx+1)
            plt.title(what)
            for name in metrics:
                plt.plot(getattr(metrics[name], f"{tip[0]}{what}"), label=name)
            plt.legend()
        plt.show()


def calcMetrics(gt,img):
    #precrec = precision_recall_fscore_support(gt.flatten(), img.flatten())
    metrics = np.zeros((7,4))
    #print((gt.shape, img.shape))
    
    for clas in range(7):
        inter = ((gt==clas)*(img==clas)).sum() #(sliceinter==clas).sum()
    
        gtclas = (gt==clas).sum()
        imclas = (img==clas).sum()
        #recall
        tmp = inter/gtclas if gtclas>0 else np.nan
        metrics[clas, 1] = tmp
        #precision
        tmr = inter/imclas if imclas>0 else np.nan
        metrics[clas, 2] = tmr 
        #fscore
        metrics[clas, 3] = np.nan if (np.isnan(tmr) or np.isnan(tmp)) else 2*tmp*tmr/(tmp+tmr)
        #dice
        metrics[clas, 0] = 2*inter/(gtclas+imclas) if (gtclas>0 or imclas>0) else np.nan
    return metrics #, precrec
 
def getAllMetrics(fajl, subjnr, slicetuple=None): #(120,35,70)):
    fil = 'out' + str(subjnr) + '_'+fajl+'.npy'

    gt = list(Path('POEM', 'segms').glob(f'Cropped*{fil[3:9]}*'))[0]
    gt = nib.load(str(gt)).get_fdata()
    img = np.load(fil)

   # print((gt.shape, img.shape))

    #crop to same size <-- shouldn't be needed anymore, as saving is fixed
    #gt, img = CenterCropTensor3d(gt, img)
    ####
    #gx,gy,gz = gt.shape
    #img = img[:gx,:gy,:gz]
    ####

   # print((gt.shape, img.shape))
    if slicetuple==None:
        metrics = calcMetrics(gt, img)
        print('\n3D METRICS')
        print('         Dice,  Recall, Precision, F1: ')
        for i, lst in enumerate(metrics):
            print(f"class {i}: {lst}")
        return {'3D': metrics}

    s1, s2, s3 = slicetuple
    metrics1 = calcMetrics(gt[s1,:,:],img[s1,:,:])
    metrics2 = calcMetrics(gt[:,s2,:],img[:,s2,:])
    metrics3 = calcMetrics(gt[:,:,s3],img[:,:,s3])

    np.set_printoptions(precision=4)

    print('\nSAGGITAL SLICE')
    print('         Dice,  Recall, Precision, F1: ')
    for i, lst in enumerate(metrics1):
        print(f"class {i}: {lst}")


    print('\nCORONAL SLICE')
    print('         Dice, Recall, Precision, F1: ')
    for i, lst in enumerate(metrics2):
        print(f"class {i}: {lst}")

    print('\nAXIAL SLICE')
    print('         Dice, Recall, Precision, F1: ')
    for i, lst in enumerate(metrics3):
        print(f"class {i}: {lst}")

    return {'sagital': metrics1, 'coronal': metrics2, 'axial': metrics3}


# %%
