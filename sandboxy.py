#%%
from Postprocessing import Compute3DDice
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nib
from helpers import CenterCropTensor3d

#Compute3DDice(500017,'poem80_dts/unet_w/unet',80)
#Compute3DDice(500017,'poem80_dts/deepmed_w_dts/deepmed_dts',80)

#Compute3DDice(500017,'poem25-3D/deepmed/deepmed',25)
#Compute3DDice(500017,'poem25_all/pnet3d/pnet3d',25)

#when step=1, cuda is a must, it's slow anyhow
#Compute3DDice(500017,'poem25_new/unet/unet',25, batch=20, step=1, dev='cuda') 
#Compute3DDice(500017,'poem25_new/unet_dts/unet_dts',25, batch=20)
#Compute3DDice(500017,'poem25_new/unet_dts/unet_dts',25, dev='cuda', step=2)


#%%
mapa = 'poem'
konc = '_v4'
Compute3DDice(500062, mapa+'/deepmed1/deepmed', 25, batch=32, step=3, dev='cuda',saveout=True, savename='dm1'+konc)
Compute3DDice(500062, mapa+'/deepmed_dts1/deepmed', 25, batch=32, step=3, dev='cuda',saveout=True, savename='dm1dts'+konc)
Compute3DDice(500062, mapa+'/pnet1/pnet', 25, batch=32, step=3, dev='cuda',saveout=True,savename='pn1'+konc)
Compute3DDice(500062, mapa+'/pnet_dts1/pnet', 25, batch=32, step=3, dev='cuda',saveout=True, savename='pn1dts'+konc)
Compute3DDice(500062, mapa+'/unet1/unet', 25, batch=32, step=3, dev='cuda',saveout=True, savename='un1'+konc)
Compute3DDice(500062, mapa+'/unet_dts1/unet', 25, batch=32, step=3, dev='cuda',saveout=True, savename='un1dts'+konc)


#%%
Compute3DDice(500062, mapa+'/deepmed2/deepmed', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='dm2'+konc)
Compute3DDice(500062, mapa+'/deepmed_dts2/deepmed', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='dm2dts'+konc)
Compute3DDice(500062, mapa+'/pnet2/pnet', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='pn2'+konc)
Compute3DDice(500062, mapa+'/pnet_dts2/pnet', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='pn2dts'+konc)
Compute3DDice(500062, mapa+'/unet2/unet', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='un2'+konc)
Compute3DDice(500062, mapa+'/unet_dts2/unet', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='un2dts'+konc)


#%%
mapa = 'poem'
konc = '_v8'
Compute3DDice(500062, mapa+'/deepmed_dts3d/deepmed', 25, batch=32, step=3, dev='cuda', saveout=True, savename='dm3Ddts'+konc)
#Compute3DDice(500062, mapa+'/deepmed3d/deepmed', 25, batch=32, step=3, dev='cuda', saveout=True, savename='dm3D'+konc)

#%%
Compute3DDice(500062, mapa+'/pnet3d/pnet', 25, batch=32, step=3, dev='cuda', saveout=True, savename='pn3D'+konc)
Compute3DDice(500062, mapa+'/pnet_dts3d/pnet', 25, batch=32, step=3, dev='cuda', saveout=True, savename='pn3Ddts'+konc)
Compute3DDice(500062, mapa+'/unet3d/unet', 25, batch=32, step=3, dev='cuda', saveout=True, savename='un3D'+konc)
Compute3DDice(500062, mapa+'/unet_dts3d/unet', 25, batch=32, step=3, dev='cuda', saveout=True, savename='un3Ddts'+konc)


#%%
#check how it looks
def plotSeg(fajl, subjnr, slicestuple=(120,35,70), patchsize=25, bydim=0):
    #fajl = 'dm1dts_v4'
    fil = 'out' + str(subjnr) + '_'+fajl+'.npy'

    #which slices to plot:
    s1,s2,s3 = slicestuple #(120,35,70)
    #bydim = 0 #set to 0 when in 3d
    #patchsize=25
    step=3
    dm = fajl[:2]=='dm'

    img = np.load(fil)

    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img[:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('axial OUT')
    plt.axis('off')
    plt.subplot(2,3,2)
    plt.imshow(img[:,s2,:].squeeze(), vmin=0, vmax=7)
    plt.title('coronal OUT')
    plt.axis('off')
    plt.subplot(2,3,3)
    plt.imshow(img[s1,:,:].squeeze(), extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('saggital OUT')
    plt.axis('off')

    gt = list(Path('POEM', 'segms').glob(f'Cropped*{fil[3:9]}*'))[0]
    gt = nib.load(str(gt)).get_fdata()
    mask = list(Path('POEM', 'masks').glob(f'cropped*{fil[3:9]}*'))[0]
    mask = nib.load(str(mask)).get_fdata()

    leavebckg = (patchsize-16*dm)//2
    x, y, z = mask.shape
    tmp = mask.sum(axis=(0,1))
    startz, endz = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
    tmp = mask.sum(axis=(1,2))
    startx, endx = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
    tmp = mask.sum(axis=(0,2))
    starty, endy = np.nonzero(tmp)[0][0], np.nonzero(tmp)[0][-1]
    startx = max(0, startx-leavebckg) 
    starty = max(0, starty-leavebckg) 
    startz = max(0, startz-leavebckg) 
    endx = min(x, endx+leavebckg+1)
    endy = min(y, endy+leavebckg+1)
    endz = min(z, endz+leavebckg+1)
    gt = gt[startx:endx, starty:endy, startz:endz]
    pad_width = [(0,patchsize)]*3
    if bydim>0: pad_width[bydim] = (0,0)
    gt = np.pad(gt, pad_width, mode='constant')

    plt.subplot(2,3,4)
    plt.imshow(gt[:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('axial GT')
    plt.axis('off')
    plt.subplot(2,3,5)
    plt.imshow(gt[:,s2,:].squeeze(), vmin=0, vmax=7)
    plt.title('coronal GT')
    plt.axis('off')
    plt.subplot(2,3,6)
    plt.imshow(gt[s1,:,:].squeeze(), extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('sagital GT')
    plt.axis('off')

#%%
#calc alco dices for every slice you show:

#from sklearn.metrics import precision_recall_fscore_support

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
 
def getAllMetrics(fajl, subjnr, slicetuple=(120,35,70)):
    fil = 'out' + str(subjnr) + '_'+fajl+'.npy'
    s1, s2, s3 = slicetuple

    gt = list(Path('POEM', 'segms').glob(f'Cropped*{fil[3:9]}*'))[0]
    gt = nib.load(str(gt)).get_fdata()
    img = np.load(fil)

    #crop to same size

    print((gt.shape, img.shape))
    gt, img = CenterCropTensor3d(gt, img)

    print((gt.shape, img.shape))
    metrics1 = calcMetrics(gt[s1,:,:],img[s1,:,:])
    metrics2 = calcMetrics(gt[:,s2,:],img[:,s2,:])
    metrics3 = calcMetrics(gt[:,:,s3],img[:,:,s3])

    np.set_printoptions(precision=3)

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





#%% # check dices on a subject 
np.set_printoptions(precision=3)
fajl = 'dm3Ddts_v7'
dcs = 'dices_'+fajl+'.npy'
dcs = np.load(dcs)
print(dcs[0,1:])

#%%
#check training/Val curves
from Postprocessing_old import compare_curves
compare_curves('poem', ['deepmed3d', 'unet3d'])


# %%
#get validation set class sizes for better sampling/loss settings:
from helpers import getpid

folder = 'POEM25'

allpids = np.unique([getpid(str(i)) for i in Path(folder, 'VAL', 'gt').glob('*.npy')])
allfiles = sorted([p for p in Path('POEM', 'segms').glob('*') if getpid(str(p)) in allpids])
allmasks = sorted([p for p in Path('POEM', 'masks').glob('*') if getpid(str(p)) in allpids])
N = len(allpids)
classes = np.zeros((7,))
for fil,mask in zip(allfiles, allmasks): 
    fajl = nib.load(str(fil)).get_fdata()
    maskca = nib.load(str(mask)).get_fdata()
    vsi = fajl[maskca==1]
    uniq, counts = np.unique(vsi, return_counts=True)
    classes[uniq.astype(int)] += counts

allinall = sum(classes)
avg_classes = classes/N
weighting = avg_classes/allinall

t=1/weighting
s=sum(t)
print(t/s) #[0.0004 , 0.217, 0.156, 0.016, 0.337, 0.113, 0.16 ]
#suggested weighting:
[0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]
# %%
