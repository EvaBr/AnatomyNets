#%%
from Postprocessing import Compute3DDice
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nib

#Compute3DDice(500017,'poem80_dts/unet_w/unet',80)
#Compute3DDice(500017,'poem80_dts/deepmed_w_dts/deepmed_dts',80)

#Compute3DDice(500017,'poem25-3D/deepmed/deepmed',25)
#Compute3DDice(500017,'poem25_all/pnet3d/pnet3d',25)

#when step=1, cuda is a must, it's slow anyhow
#Compute3DDice(500017,'poem25_new/unet/unet',25, batch=20, step=1, dev='cuda') 
#Compute3DDice(500017,'poem25_new/unet_dts/unet_dts',25, batch=20)
#Compute3DDice(500017,'poem25_new/unet_dts/unet_dts',25, dev='cuda', step=2)

#%%
print(Compute3DDice(500062, 'poem25/deepmed1/deepmed', 25, batch=20, step=3, dev='cuda',saveout=True, savename='dm1'))
print(Compute3DDice(500062, 'poem25/deepmed_dts1/deepmed', 25, batch=20, step=3, dev='cuda',saveout=True, savename='dm1dts'))
print(Compute3DDice(500062, 'poem25/pnet1/pnet', 25, batch=20, step=3, dev='cuda',saveout=True,savename='pn1'))
print(Compute3DDice(500062, 'poem25/pnet_dts1/pnet', 25, batch=20, step=3, dev='cuda',saveout=True, savename='pn1dts'))
print(Compute3DDice(500062, 'poem25/unet1/unet', 25, batch=20, step=3, dev='cuda',saveout=True, savename='un1'))
print(Compute3DDice(500062, 'poem25/unet_dts1/unet', 25, batch=20, step=3, dev='cuda',saveout=True, savename='un1dts'))


#%%
Compute3DDice(500062, 'poem25/deepmed2/deepmed', 25, batch=20, bydim=2, step=3, dev='cuda', saveout=True, savename='dm2')
Compute3DDice(500062, 'poem25/deepmed_dts2/deepmed', 25, batch=20, bydim=2, step=3, dev='cuda', saveout=True, savename='dm2dts')
Compute3DDice(500062, 'poem25/pnet2/pnet', 25, batch=20, bydim=2, step=3, dev='cuda', saveout=True, savename='pn2')
Compute3DDice(500062, 'poem25/pnet_dts2/pnet', 25, batch=20, bydim=2, step=3, dev='cuda', saveout=True, savename='pn2dts')
Compute3DDice(500062, 'poem25/unet2/unet', 25, batch=20, bydim=2, step=3, dev='cuda', saveout=True, savename='un2')
Compute3DDice(500062, 'poem25/unet_dts2/unet', 25, batch=20, bydim=2, step=3, dev='cuda', saveout=True, savename='un2dts')


#%%
Compute3DDice(500062, 'poem25/deepmed3d/deepmed', 25, batch=10, step=3, dev='cuda', saveout=True, savename='dm3D')
Compute3DDice(500062, 'poem25/deepmed_dts3d/deepmed', 25, batch=10, step=3, dev='cuda', saveout=True, savename='dm3Ddts')
Compute3DDice(500062, 'poem25/pnet3d/pnet', 25, batch=10, step=3, dev='cuda', saveout=True, savename='un3D')
Compute3DDice(500062, 'poem25/pnet_dts3d/pnet', 25, batch=10, step=3, dev='cuda', saveout=True, savename='un3Ddts')
Compute3DDice(500062, 'poem25/unet3d/unet', 25, batch=10, step=3, dev='cuda', saveout=True, savename='pn3D')
Compute3DDice(500062, 'poem25/unet_dts3d/unet', 25, batch=10, step=3, dev='cuda', saveout=True, savename='pn3Ddts')


#%%
#check how it looks
fajl = 'dm1dts'
fil = 'out500022_'+fajl+'.npy'
#which slices to plot:
s1,s2,s3 = (140,30,60)
bydim = 1 #set to 0 when in 3d
patchsize=25
step=3
dm = fajl[:2]=='dm'

img = np.load(fil)

plt.figure()
plt.subplot(2,3,1)
plt.imshow(img[:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(img[:,s2,:].squeeze(), vmin=0, vmax=7)
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(img[s1,:,:].squeeze(), extent=(0,100,0,150), vmin=0, vmax=7)
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
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(gt[:,s2,:].squeeze(), vmin=0, vmax=7)
plt.axis('off')
plt.subplot(2,3,6)
plt.imshow(gt[s1,:,:].squeeze(), extent=(0,100,0,150), vmin=0, vmax=7)
plt.axis('off')

#%%
#calc alco dices for every slice you show:

#from sklearn.metrics import precision_recall_fscore_support

def calcMetrics(gt,img):
    #precrec = precision_recall_fscore_support(gt.flatten(), img.flatten())
    metrics = np.zeros((7,4))
    sliceinter = (gt==img)*gt
    for clas in range(7):
        inter = ((gt==clas)*(img==clas)).sum() #(sliceinter==clas).sum()
        #assert inter == ((gt==clas)*(img==clas)).sum(), (inter, ((gt==clas)*(img==clas)).sum())
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
 
 
metrics1 = calcMetrics(gt[s1,:,:],img[s1,:,:])
metrics2 = calcMetrics(gt[:,s2,:],img[:,s2,:])
metrics3 = calcMetrics(gt[:,:,s3],img[:,:,s3])

np.set_printoptions(precision=3)
print('Dice,  Recall, Precision, F1: ')
print(metrics1)

print('\nDice, Recall, Precision, F1: ')
print(metrics2)

print('\nDice, Recall, Precision, F1: ')
print(metrics3)



# %%
fajl = 'un3Ddts'
dcs = 'dices_'+fajl+'.npy'
dcs = np.load(dcs)
print(dcs[0,1:])
# %%
